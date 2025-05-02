import json
import random
import torch
import torch.nn as nn
import torch.optim as optim
from model import DigitToVector as Model
import os
import gc

version = open("models/version.txt").read()
name = f'models/V{version}/trained_model.pth'

def generate_random_data(num_samples=100, min_size=3, max_size=10):
    input_arrays = []
    target_arrays = []
    
    for _ in range(num_samples):
        input_size = torch.randint(min_size, max_size, (2,))
        target_size = torch.randint(min_size, max_size, (2,))
        
        input_array = torch.randint(0, 9, (input_size[0], input_size[1]), dtype=torch.float32)
        target_array = torch.randint(0, 9, (target_size[0], target_size[1]), dtype=torch.float32)
        
        input_arrays.append(input_array)
        target_arrays.append(target_array)
    
    return input_arrays, target_arrays

def get_arc_data(num_samples=999):
    input_arrays = []
    target_arrays = []
    main_input_arrays = []
    desired_arrays = []
    data = json.load(open("arc-prize-2025/arc-agi_training_challenges_resampled.json"))

    for _ in range(num_samples):
        try:
            sample_data = data[_]
            infer_data = sample_data[0][0]
            
            input_array = torch.tensor(infer_data[0], dtype=torch.float32) / 9.0
            target_array = torch.tensor(infer_data[1], dtype=torch.float32) / 9.0
            main_input_array = torch.tensor(sample_data[1][0], dtype=torch.float32) / 9.0
            desired_array = torch.tensor(sample_data[1][1], dtype=torch.float32) / 9.0
            
            if (input_array.shape[0] >= 5 and input_array.shape[1] >= 5 and
                target_array.shape[0] >= 5 and target_array.shape[1] >= 5 and
                main_input_array.shape[0] >= 5 and main_input_array.shape[1] >= 5 and
                torch.all(input_array >= 0) and torch.all(input_array <= 1) and
                torch.all(target_array >= 0) and torch.all(target_array <= 1) and
                torch.all(main_input_array >= 0) and torch.all(main_input_array <= 1)):
                
                input_arrays.append(input_array)
                target_arrays.append(target_array)
                main_input_arrays.append(main_input_array)
                desired_arrays.append(desired_array)
        except Exception as e:
            print(f"Warning: Skipping sample {_}: {str(e)}")
            continue

    while len(input_arrays) < num_samples:
        try:
            size = random.randint(5, 10)
            input_array = torch.rand(size, size)
            target_array = torch.rand(size, size)
            main_input_array = torch.rand(size, size)
            desired_array = torch.rand(size, size)
            
            input_arrays.append(input_array)
            target_arrays.append(target_array)
            main_input_arrays.append(main_input_array)
            desired_arrays.append(desired_array)
        except Exception as e:
            print(f"Warning: Error generating random sample: {str(e)}")
            continue

    print(f"Generated {len(input_arrays)} valid training samples")
    return input_arrays, target_arrays, main_input_arrays, desired_arrays

def custom_collate(batch):
    return batch

def create_dataloader(input_arrays, target_arrays, main_input_arrays, desired_arrays, batch_size=32):
    valid_samples = []
    for i in range(len(input_arrays)):
        try:
            if (input_arrays[i].shape[0] >= 2 and input_arrays[i].shape[1] >= 2 and
                target_arrays[i].shape[0] >= 2 and target_arrays[i].shape[1] >= 2 and
                main_input_arrays[i].shape[0] >= 2 and main_input_arrays[i].shape[1] >= 2 and
                torch.all(input_arrays[i] >= 0) and torch.all(input_arrays[i] <= 1) and
                torch.all(target_arrays[i] >= 0) and torch.all(target_arrays[i] <= 1) and
                torch.all(main_input_arrays[i] >= 0) and torch.all(main_input_arrays[i] <= 1)):
                
                valid_samples.append((
                    input_arrays[i],
                    target_arrays[i],
                    main_input_arrays[i],
                    desired_arrays[i]
                ))
        except Exception as e:
            print(f"Warning: Skipping sample {i} in dataloader: {str(e)}")
            continue
    
    if not valid_samples:
        raise ValueError("No valid samples found after filtering")
    
    print(f"Created dataloader with {len(valid_samples)} valid samples")
    return torch.utils.data.DataLoader(
        valid_samples, 
        batch_size=batch_size, 
        shuffle=True,
        collate_fn=custom_collate
    )

def compute_loss(prediction, target):
    pred_size = prediction.shape
    target_size = target.shape
    
    padded_pred = torch.zeros_like(target)
    
    min_rows = min(pred_size[0], target_size[0])
    min_cols = min(pred_size[1], target_size[1])
    padded_pred[:min_rows, :min_cols] = prediction[:min_rows, :min_cols]
    
    return nn.MSELoss()(padded_pred, target)

def train_model(model, train_dataloader, num_epochs=10, learning_rate=0.001):
    global version
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print(f"Starting training with {len(train_dataloader)} batches per epoch")
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        batch_count = 0
        valid_batch_count = 0
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        for batch_idx, batch in enumerate(train_dataloader):
            try:
                optimizer.zero_grad()
                batch_loss = 0
                valid_samples = 0
                
                for inputs, targets, main_inputs, desired_outputs in batch:
                    try:
                        outputs = model.forward(inputs, targets, main_inputs)
                        loss = compute_loss(outputs, desired_outputs)
                        batch_loss += loss
                        valid_samples += 1
                    except ValueError as e:
                        continue
                    except Exception as e:
                        print(f"Warning: Error processing sample: {str(e)}")
                        continue
                
                if valid_samples > 0:
                    batch_loss = batch_loss / valid_samples
                    
                    batch_loss.backward()
                    optimizer.step()
                    
                    total_loss += batch_loss.item()
                    valid_batch_count += 1
                    
                    if batch_idx % 5 == 0:
                        print(f'Batch: {batch_idx}/{len(train_dataloader)}, Loss: {batch_loss.item():.4f}, Valid samples: {valid_samples}')
            except Exception as e:
                print(f"Error in batch {batch_idx}: {str(e)}")
                continue

        if not os.path.exists('models/V' + str(version)):
            os.makedirs('models/V' + str(version))

        torch.save(model.state_dict(), 'models/V' + str(version) + f'/mid_training_model_{epoch}.pth')
        print("\nModel saved as 'models/V" + str(version) + f"/mid_training_model_{epoch}.pth'")
        
        if valid_batch_count > 0:
            avg_loss = total_loss / valid_batch_count
            print(f'Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}, Valid batches: {valid_batch_count}/{len(train_dataloader)}')
        else:
            print(f'Epoch {epoch+1} completed with no successful batches')

def eval_model(model, file_name):
    global version
    try:
        print("Loading data and model...")
        data = json.load(open('arc-prize-2025/arc-agi_evaluation_challenges_resampled.json'))
        model.load_state_dict(torch.load(file_name))
        print(f"Loaded {len(data)} samples")
    except Exception as e:
        print(f"Error during initialization: {str(e)}")
        return

    error = 0
    precentage = 0
    precentcount = 0
    processed_samples = 0
    batch_size = 5  # Reduced batch size
    real_percentage = 0
    
    print(f"Starting evaluation with batch size {batch_size}")
    
    for i in range(0, len(data), batch_size):
        try:
            batch_end = min(i + batch_size, len(data))
            print(f"\nProcessing batch {i//batch_size + 1}/{(len(data) + batch_size - 1)//batch_size}")
            print(f"Sample range: {i} to {batch_end-1}")
            
            batch_error = 0
            batch_precentage = 0
            batch_precentcount = 0
            batch_processed = 0
            batch_real_percentage = 0
            
            for j in range(i, batch_end):
                try:
                    sample_data = data[j]
                    infer_data = sample_data[0][0]
                    
                    # Process input data in smaller chunks
                    input_array = torch.tensor(infer_data[0], dtype=torch.float32) / 9.0
                    target_array = torch.tensor(infer_data[1], dtype=torch.float32) / 9.0
                    main_input_array = torch.tensor(sample_data[1][0], dtype=torch.float32) / 9.0
                    desired_output = torch.tensor(sample_data[1][1], dtype=torch.float32) / 9.0
                    
                    if (input_array.shape[0] < 2 or input_array.shape[1] < 2 or
                        target_array.shape[0] < 2 or target_array.shape[1] < 2 or
                        main_input_array.shape[0] < 2 or main_input_array.shape[1] < 2):
                        print(f"Skipping sample {j}: Arrays too small")
                        del input_array, target_array, main_input_array, desired_output
                        gc.collect()
                        continue
                    
                    # Process prediction
                    with torch.no_grad():
                        prediction = model.forward(input_array, target_array, main_input_array)
                    
                    pred_size = prediction.shape
                    target_size = desired_output.shape
                    
                    # Create padded prediction
                    padded_pred = torch.zeros_like(desired_output)
                    min_rows = min(pred_size[0], target_size[0])
                    min_cols = min(pred_size[1], target_size[1])
                    padded_pred[:min_rows, :min_cols] = prediction[:min_rows, :min_cols]
                    
                    # Compute metrics
                    with torch.no_grad():
                        batch_precentage += torch.sum(torch.round(padded_pred) == desired_output)
                        batch_precentcount += prediction.shape[0] * prediction.shape[1]
                        batch_error += nn.MSELoss()(padded_pred, desired_output)
                        batch_real_percentage += 1 if (torch.sum(torch.round(padded_pred) == desired_output) == prediction.shape[0] * prediction.shape[1]) else 0

                    batch_processed += 1
                    
                    # Clean up tensors
                    del input_array, target_array, main_input_array, desired_output, prediction, padded_pred
                    gc.collect()
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    
                except Exception as e:
                    print(f"Error in sample {j}: {str(e)}")
                    gc.collect()
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    continue
            
            if batch_processed > 0:
                print(f"\nBatch {i//batch_size + 1} summary:")
                print(f"Processed {batch_processed} samples")
                
                error += batch_error
                precentage += batch_precentage
                precentcount += batch_precentcount
                processed_samples += batch_processed
                real_percentage += batch_real_percentage

                print(f"Running totals - Error: {error/processed_samples:.2%}, "
                      f"Accuracy: {precentage/precentcount:.2%}, "
                      f"Real accuracy: {batch_real_percentage/batch_size:.2%}")
            
            # Clean up batch tensors
            del batch_error, batch_precentage, batch_precentcount
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        except Exception as e:
            print(f"Error in batch {i//batch_size + 1}")
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            continue
            
    if processed_samples > 0:
        print("\nFinal Results:")
        print(f"Total samples processed: {processed_samples}")
        print(f"Final error: {error/processed_samples::.2%}")
        print(f"Final accuracy: {precentage/precentcount::.2%}")
        print(f"Final real accuracy: {real_percentage/processed_samples:.2%}")
    else:
        print("No valid samples were processed")

def make_model(model, file_name = None):
    global version
    print("Generating training data...")
    input_arrays, target_arrays, main_input_arrays, desired_arrays = get_arc_data()
    print(f"Generated {len(input_arrays)} training samples")
    train_dataloader = create_dataloader(input_arrays, target_arrays, main_input_arrays, desired_arrays, batch_size=10)

    model = Model()
    if file_name is not None:
        try:
            state_dict = torch.load(file_name)
            model_state_dict = model.state_dict()
            if set(state_dict.keys()) == set(model_state_dict.keys()):
                model.load_state_dict(state_dict)
                print(f"Successfully loaded model from {file_name}")
            else:
                open('models/version.txt', 'w').write(f'{version.split('.')[0]}.{version.split('.')[1]}.{version.split('.')[2] + 1}')
                version = open("models/version.txt").read()
                print(f"Warning: Model architecture has changed. Starting with fresh model.")
        except Exception as e:
            print(f"Warning: Could not load model from {file_name}: {str(e)}")
            print("Starting with fresh model.")
    
    if torch.cuda.is_available():
        model.cuda()

    train_model(model, train_dataloader, num_epochs=10)
    
    torch.save(model.state_dict(), 'models/V' + str(version) + '/trained_model.pth')
    print("\nModel saved as '/models/V" + str(version) + "/trained_model.pth'")
    
    print("\nTesting model with random arrays...")
    model.eval()
    with torch.no_grad():
        try:
            test_input = input_arrays[random.randint(0, len(input_arrays) - 1)]
            test_output = target_arrays[random.randint(0, len(target_arrays) - 1)]
            test_main_input = main_input_arrays[random.randint(0, len(main_input_arrays) - 1)]
            
            prediction = model(test_input, test_output, test_main_input)
            print("Test Error:", compute_loss(prediction, test_output))
        except Exception as e:
            print(f"Error during testing: {str(e)}")

if __name__ == "__main__":
    #make_model(Model(), name)
    eval_model(Model(), name)