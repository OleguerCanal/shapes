from data_gathering.control_robot import ControlRobot
from data_gathering.data_collector import DataCollector


if __name__ == "__main__":

    experiment_name = "air_test"
    movement_list = [
        [-10, 0, 0],
        [-10, 0, 0]#,
        # [0, 0, 10],
        # [10, 0, 0],
        # [10, 0, 0],
        # [0, 0, 10],
        # [-10, 0, 0],
        # [-10, 0, 0]
    ]

    cr = ControlRobot()
    cr.perfrom_experiment(
        experiment_name=experiment_name,
        movement_list=movement_list
    )
