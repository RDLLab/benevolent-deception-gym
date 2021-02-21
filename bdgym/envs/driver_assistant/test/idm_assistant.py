"""Run autopilot_env with IDM autopilot.

I.e. "autopilot" simply returns actual observations of the
controlled vehicle and actions calculated by IDMVehicle
"""
import bdgym.envs.driver_assistant.test.test_utils as test_utils


if __name__ == "__main__":
    parser = test_utils.test_parser()
    args = test_utils.parse_parser(parser)
    test_utils.run(args)
