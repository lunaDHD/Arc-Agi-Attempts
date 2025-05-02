import torch
import torch.nn as nn
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor
import numpy as np

class AttentionLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AttentionLayer, self).__init__()
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)
        self.scale = torch.sqrt(torch.FloatTensor([hidden_dim]))
        
    def forward(self, query, key, value):
        Q = self.query(query)  # [batch_size, hidden_dim]
        K = self.key(key)      # [batch_size, hidden_dim]
        V = self.value(value)  # [batch_size, hidden_dim]
        
        attention = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attention = F.softmax(attention, dim=-1)
        
        output = torch.matmul(attention, V)
        return output

class DigitToVector(nn.Module):
    def __init__(self):
        super(DigitToVector, self).__init__()
        self.encoder2x2 = nn.Sequential(
            nn.Linear(2*2*2, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 256),
            nn.Tanh(),
        )
        self.encoder3x3 = nn.Sequential(
            nn.Linear(3*3*2, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 256),
            nn.Tanh(),
        )
        self.encoder4x4 = nn.Sequential(
            nn.Linear(4*4*2, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 256),
            nn.Tanh(),
        )
        
        self.attentionlst = [AttentionLayer(256, 256) for _ in range(5)]
        
        self.decoder = nn.Sequential(
            nn.Linear(256 + 9, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 9),
            nn.Sigmoid()
        )
    
    def process_windows(self, input_array, output_array):
        n = input_array.shape
        m = output_array.shape

        if n[0] < 2 or n[1] < 2 or m[0] < 2 or m[1] < 2:
            raise ValueError("Input arrays must be at least 2x2")

        input_windows = {
            2: [],
            3: [],
            4: []
        }
        output_windows = {
            2: [],
            3: [],
            4: []
        }
        
        for i in range(n[0]-1):
            for j in range(n[1]-1):
                window = input_array[i:i+2, j:j+2]
                window = torch.flatten(window)
                input_windows[2].append(window)
        
        for i in range(m[0]-1):
            for j in range(m[1]-1):
                window = output_array[i:i+2, j:j+2]
                window = torch.flatten(window)
                output_windows[2].append(window)

        if n[0] >= 3 and n[1] >= 3 and m[0] >= 3 and m[1] >= 3:
            for i in range(n[0]-2):
                for j in range(n[1]-2):
                    window = input_array[i:i+3, j:j+3]
                    window = torch.flatten(window)
                    input_windows[3].append(window)
            
            for i in range(m[0]-2):
                for j in range(m[1]-2):
                    window = output_array[i:i+3, j:j+3]
                    window = torch.flatten(window)
                    output_windows[3].append(window)

        if n[0] >= 4 and n[1] >= 4 and m[0] >= 4 and m[1] >= 4:
            for i in range(n[0]-3):
                for j in range(n[1]-3):
                    window = input_array[i:i+4, j:j+4]
                    window = torch.flatten(window)
                    input_windows[4].append(window)
            
            for i in range(m[0]-3):
                for j in range(m[1]-3):
                    window = output_array[i:i+4, j:j+4]
                    window = torch.flatten(window)
                    output_windows[4].append(window)

        encoded_vectors = []
        
        if len(input_windows[2]) > 0 and len(output_windows[2]) > 0:
            try:
                input_windows_tensor = torch.stack(input_windows[2])
                output_windows_tensor = torch.stack(output_windows[2])
                
                n_input = len(input_windows_tensor)
                n_output = len(output_windows_tensor)
                
                input_expanded = input_windows_tensor.unsqueeze(1).expand(-1, n_output, -1)
                output_expanded = output_windows_tensor.unsqueeze(0).expand(n_input, -1, -1)
                
                combined = torch.cat([input_expanded, output_expanded], dim=2)
                combined = combined.view(-1, combined.size(-1))
                
                encoded = self.encoder2x2(combined)
                encoded = encoded.view(n_input, n_output, 256)
                encoded_vectors.append(encoded.mean(dim=(0, 1)))
            except Exception as e:
                print(f"Warning: Error processing 2x2 windows: {str(e)}")
        
        if len(input_windows[3]) > 0 and len(output_windows[3]) > 0:
            try:
                input_windows_tensor = torch.stack(input_windows[3])
                output_windows_tensor = torch.stack(output_windows[3])
                
                n_input = len(input_windows_tensor)
                n_output = len(output_windows_tensor)
                
                input_expanded = input_windows_tensor.unsqueeze(1).expand(-1, n_output, -1)
                output_expanded = output_windows_tensor.unsqueeze(0).expand(n_input, -1, -1)
                
                combined = torch.cat([input_expanded, output_expanded], dim=2)
                combined = combined.view(-1, combined.size(-1))
                
                encoded = self.encoder3x3(combined)
                encoded = encoded.view(n_input, n_output, 256)
                encoded_vectors.append(encoded.mean(dim=(0, 1)))
            except Exception as e:
                print(f"Warning: Error processing 3x3 windows: {str(e)}")
        
        if len(input_windows[4]) > 0 and len(output_windows[4]) > 0:
            try:
                input_windows_tensor = torch.stack(input_windows[4])
                output_windows_tensor = torch.stack(output_windows[4])
                
                n_input = len(input_windows_tensor)
                n_output = len(output_windows_tensor)
                
                input_expanded = input_windows_tensor.unsqueeze(1).expand(-1, n_output, -1)
                output_expanded = output_windows_tensor.unsqueeze(0).expand(n_input, -1, -1)
                
                combined = torch.cat([input_expanded, output_expanded], dim=2)
                combined = combined.view(-1, combined.size(-1))
                
                encoded = self.encoder4x4(combined)
                encoded = encoded.view(n_input, n_output, 256)
                encoded_vectors.append(encoded.mean(dim=(0, 1)))
            except Exception as e:
                print(f"Warning: Error processing 4x4 windows: {str(e)}")
        
        if not encoded_vectors:
            raise ValueError("No valid windows could be processed")
        
        # Stack encoded vectors and apply attention between them
        attended = torch.stack(encoded_vectors)  # [num_vectors, 256]
        
        # Apply self-attention between encoded vectors
        for i in self.attentionlst:
            attended = i(attended, attended, attended)
        
        # Take mean of attended vectors
        final_vector = attended.mean(dim=0)
        
        return final_vector
    
    def process_main_windows(self, main_input_array, encoded_vector):
        windows = []
        positions = []

        for i in range(main_input_array.shape[0]-2):
            for j in range(main_input_array.shape[1]-2):
                window = main_input_array[i:i+3, j:j+3]
                window = torch.flatten(window)
                windows.append(window)
                positions.append((i, j))
        
        if not windows:
            raise ValueError("No valid windows could be processed")
            
        windows = torch.stack(windows)
        encoded_expanded = encoded_vector.unsqueeze(0).expand(len(windows), -1)
        combined = torch.cat([encoded_expanded, windows], dim=1)
        decoded = self.decoder(combined)
        
        final_output = torch.zeros_like(main_input_array)
        for idx, (i, j) in enumerate(positions):
            final_output[i:i+3, j:j+3] = decoded[idx].view(3, 3)
        
        return final_output
        
    def forward(self, input_array, output_array, main_input_array):
        encoded_vector = self.process_windows(input_array, output_array)
        return self.process_main_windows(main_input_array, encoded_vector)

if __name__ == "__main__":
    model = DigitToVector()
    
    input_array1 = torch.rand(5, 5)
    input_array2 = torch.rand(4, 4)
    main_input = torch.rand(6, 6)
    
    if input_array1.shape[0] < 3 or input_array1.shape[1] < 3:
        raise ValueError("input_array1 must be at least 3x3")
    if input_array2.shape[0] < 3 or input_array2.shape[1] < 3:
        raise ValueError("input_array2 must be at least 3x3")
    if main_input.shape[0] < 3 or main_input.shape[1] < 3:
        raise ValueError("main_input must be at least 3x3")
    
    output = model(input_array1, input_array2, main_input)
    print("Input shapes:", input_array1.shape, input_array2.shape, main_input.shape)
    print("Output shape:", output.shape)
    print("Output range:", output.min().item(), "to", output.max().item())
