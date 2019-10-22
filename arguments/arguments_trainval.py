from .arguments_base import ArgumentsBase


class ArgumentsTrainVal(ArgumentsBase):
    def __init__(self):
        super(ArgumentsTrainVal, self).__init__()

        parser = self.parser
        # model save info
        parser.add_argument('-cifrec', '--checkpoint-iter-freq', default=1000, type=int,
                            help='the frequency of saving model under iteration')
        parser.add_argument('-cefrec', '--checkpoint-epoch-freq', default=5, type=int,
                            help='the frequency of saving model under epoch')
        parser.add_argument('-cpath', '--checkpoint_save-path',
                            required=True, help='the directory to save model')

        # training control parameters
        parser.add_argument('-e', '--maxepoch', required=True,
                            type=int, help='the number of epochs to train')
        parser.add_argument('-lr', '--learning-rate', default=0.1,
                            type=float, help='initial learning rate')
        parser.add_argument('-m', '--momentum', default=0.9,
                            type=float, help='momentum')
        parser.add_argument('-wd', '--weight-decay', default=1e-4,
                            type=float, help='weight decay(L1 penalty)')
        parser.add_argument('-cg', '--cumulate-gradient', default=1,
                            type=int, help='cumulate gradient for large network')

        # model info
        parser.add_argument('-rm', '--resume-model', default=None,
                            help='resume model file', metavar='FILE')
        parser.add_argument('-iter', '--resume-iteration',
                            default=0, type=int, help='resume iteration number')
        parser.add_argument('-epo', '--resume-epoch', default=0,
                            type=int, help='resume epoch number')

        # model boosting info
        parser.add_argument('-ada', '--ada-train', default=False,
                            type=bool, help='move to next round of the boosting')
        parser.add_argument('-lam', '--lambda-error', default=0.7,
                            type=float, help='lamba controls the weight differences')
