from ImgTrainer import make_model, eval_model, name
from model import DigitToVector as Model

make_model(Model(), name)
eval_model(Model(), name)
