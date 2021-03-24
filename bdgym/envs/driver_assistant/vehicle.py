"""Vehicle class for Driver Assistant Environment """
from highway_env.vehicle.kinematics import Vehicle


class DriverAssistantVehicle(Vehicle):
    """A faster vehicle """

    MAX_SPEED = 40.
