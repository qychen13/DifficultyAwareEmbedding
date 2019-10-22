from .arguments_base import ArgumentsBase


class ArgumentsTest(ArgumentsBase):
    def __init__(self):
        super(ArgumentsTest, self).__init__()

        parser = self.parser
        # model info
        parser.add_argument('-tm', '--test-models', default=None,
                            help='test model files', metavar='FILE', nargs='+')
