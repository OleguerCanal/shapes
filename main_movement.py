from data_gathering.control_robot import ControlRobot
from data_gathering.data_collector import DataCollector


if __name__ == "__main__":

    experiment_name = "datasets/demo_video"
    d = 35
    dv = 15
    movement_list = [
        [-d, 0, 0],
        [-d, 0, 0],
        [0, 0, dv],
        [d, 0, 0],
        [d, 0, 0],
        [0, 0, dv],
        [-d, 0, 0],
        [-d, 0, 0]
    ]

    cr = ControlRobot()
    cr.perfrom_experiment(
        experiment_name=experiment_name,
        movement_list=movement_list
    )
