"""Run script for the Exercise Assistant Environment """

import bdgym.scripts.exercise_assistant.utils as utils


if __name__ == "__main__":
    parser = utils.argument_parser()
    utils.run(utils.parse_parser(parser))
