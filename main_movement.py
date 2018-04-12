from gathering.control_robot import ControlRobot
from gathering.data_collector import DataCollector


if __name__ == "__main__":

    experiment_name = "datasets/stitching_dataset"
    d = 5
    movement_list = [
        [-d, 0, 0],
        [-d, 0, 0],
        [-d, 0, 0],
        [-d, 0, 0],
        [-d, 0, 0],
        [-d, 0, 0],
        [-d, 0, 0],
        [-d, 0, 0],
        [-d, 0, 0],
        [-d, 0, 0],
        [-d, 0, 0],
        [-d, 0, 0],
        [-d, 0, 0],
        [-d, 0, 0],
        [-d, 0, 0]
    ]

    cr = ControlRobot()
    cr.perfrom_experiment(
        experiment_name=experiment_name,
        movement_list=movement_list
    )
