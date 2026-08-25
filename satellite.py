import numpy as np

'''
Satellite pass geometry: orbital / angular speed, apparent slew rate and
point-ahead angle for a ground station tracking a satellite.
'''

grav_const = 6.674e-11   # gravitational constant [m^3 kg^-1 s^-2]
earth_mass = 5.972e24    # mass of the Earth [kg]
earth_radius = 6371e3    # mean radius of the Earth [m]
c = 2.998e8              # speed of light [m/s]


class Satellite:
    """A satellite in a circular orbit at a given altitude."""

    def __init__(self, altitude):
        """
        Parameters:
            altitude : float
                Orbital altitude above the Earth's surface [m].
        """
        self.altitude = altitude
        self.orbital_speed = np.sqrt(
            grav_const * earth_mass / (earth_radius + self.altitude))

    def angular_speed(self):
        """
        Angular velocity of the satellite about the Earth's centre.

        Returns:
            float
                Angular speed [deg/s].
        """
        angular_speed = self.orbital_speed / (earth_radius + self.altitude)
        return np.rad2deg(angular_speed)


class SatellitePass:
    """A pass of a Satellite over a ground station at a given elevation."""

    def __init__(self, satellite, elevation):
        """
        Parameters:
            satellite : Satellite
                The satellite being tracked.
            elevation : float
                Elevation angle of the satellite above the horizon [deg].
        """
        self.elevation = elevation
        self.satellite = satellite

    def tangential_velocity(self):
        """
        Component of the satellite's orbital velocity transverse to the
        line of sight, as seen from the ground station.

        Returns:
            float
                Tangential velocity [m/s].
        """
        return self.satellite.orbital_speed * np.sin(np.radians(self.elevation))

    def slant_range(self):
        """
        Slant range from the ground station to the satellite.

        Returns:
            float
                Slant range [m].
        """
        Re = earth_radius
        h = self.satellite.altitude
        el = np.radians(self.elevation)
        return -Re * np.sin(el) + np.sqrt(Re**2 * np.sin(el)**2 + h**2 + 2 * Re * h)

    def point_ahead_angle(self):
        """
        Point-ahead angle (PAA): the angular lead required to account for the
        finite speed of light over the round trip.

        Returns:
            float
                Point-ahead angle [rad].
        """
        return 2 * self.tangential_velocity() / c

    def apparent_slew_rate(self):
        """
        Apparent angular slew rate of the line of sight required to track the
        satellite.

        Returns:
            float
                Apparent slew rate [deg/s].
        """
        return np.rad2deg(self.tangential_velocity() / self.satellite.altitude)
