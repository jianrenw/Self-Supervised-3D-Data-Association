from triplet_trainer import TripletTrainer
from options import Config


def train(opt):
    trainer = TripletTrainer(opt)
    trainer.train()


if __name__ == "__main__":
    opt = Config().parse()
    print(opt)
    train(opt)
