from argparse import ArgumentParser
from AI_code_challenge.classifier.train import Train
from AI_code_challenge.classifier.test import Test

class TrainExecutor():
    def __init__(self, config):
        self.config = config

    def execute(self):
        Train(self.config)

class TestExecutor():
    def __init__(self, config):
        self.config = config

    def execute(self):
        Test(self.config)

if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("--config", help="Location of the configuration file")
    argparser.add_argument("--mode", help = "Mode of operation: train, finetune or test")
    
    # Get dictionary with runtime arguments
    args = vars(argparser.parse_args())     # Dictionary with arguments
    assert args["config"] != None, "A valid configuation file is required"

    # Execute specific blocks based on operating mode
    if args['mode'] == 'Train':
        trainer = TrainExecutor(args['config'])
        trainer.execute()
    elif args['mode'] == 'Test':
        tester = TestExecutor(args['config'])
        tester.execute()
    