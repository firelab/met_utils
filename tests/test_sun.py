"""Tests code related to predicting sun position from [1]

[1] Meeus, Jean. 1998. Astronomical Algorithms. 2nd edition. Richmond, Va: Willmann-Bell.
"""

import unittest
import sun
import astropy.units as u
import astropy.coordinates as c
import astropy.time as t
import numpy as np

class TestEcliptic(unittest.TestCase) : 
    """Exercises the Ecliptic class using test case data from [1]
    
    In each test, the computed values are tested to match within 1/2 of 
    the least significant digit in the reported value.
    """
    def setUp(self) : 
        self.td = t.Time("1987-04-10", scale='tt')
        
    def test_mean_obliquity(self) : 
        """Test mean olbiquity computation
        
        Test data from example 22a on p 148"""
        e = sun.Ecliptic(self.td)
        me = e.get_mean_obliquity()
        self.assertLess(np.abs(me - c.Angle( (23,26,27.407), unit=u.deg)), 0.0005*u.arcsec)
        
    def test_true_obliquity(self) : 
        """Test true olbiquity computation (hence nutation as well)
        
        Test data from example 25a on p 165, since nutation is implemented using
        approximation in Equation 25.8"""
        e = sun.Ecliptic(t.Time('1992-10-13', scale='tt'))
        te = e.get_true_obliquity()
        self.assertLess(np.abs(te - 23.43999*u.deg), 0.000005*u.deg)
        
    def test_moon_asc_lon(self) : 
        """Test longitude of the ascending node of the Moon's mean orbit on the ecliptic
        
        Test data from example 22a on p 148"""
        e = sun.Ecliptic(self.td)
        moon_lon = e.get_moon_asc_lon()
        self.assertLess(np.abs(moon_lon - c.Angle(371.2531*u.deg)), 0.00005*u.deg)
        
    def test_nutation_obliquity(self): 
        """Test computation of nutation in obliquity
        
        Test data from example 25.a on p. 165, due to the fact that the class
        implements the approximation in equation 25.8, p. 165
        """
        td = t.Time('1992-10-13', scale='tt')
        e = sun.Ecliptic(td)
        delta_eps = e.get_nutation_obliquity()
        delta_eps_given = 23.43999 * u.deg - c.Angle( (23, 26, 24.83), unit=u.deg)
        self.assertLess(np.abs(delta_eps - delta_eps_given), 0.000005*u.deg)

class TestSunPosition(unittest.TestCase)  : 
    def setUp(self) :
        """Data here come from example 25a, p165""" 
        self.t = t.Time('1992-10-13', scale='tt')
        self.pos = sun.SunPosition(self.t)
        
    def test_julian_century(self) : 
        """basic test of conversion to julian centuries"""
        jc = sun.jc(self.t)
        self.assertLess(np.abs(jc - (-0.072183436 * sun.julian_century)), 0.5e-9 * sun.julian_century)
        
    def test_geometric_mean_lon(self) :
        """tests geometric mean lon calculation"""
        L0 = self.pos.get_mean_lon() 
        self.assertLess(np.abs(L0 - c.Angle(-2318.19280*u.deg)), 0.5e-5*u.deg)
        
    def test_mean_anomaly(self) : 
        """tests the geometric mean anomaly calculation"""
        M = self.pos.get_mean_anomaly()
        self.assertLess(np.abs(M - c.Angle(-2241.00603*u.deg)), 0.5e-5*u.deg)
        
    def test_eoc(self) : 
        """tests the equation of center calculations"""
        eoc = self.pos.get_eoc() 
        self.assertLess(np.abs(eoc- c.Angle(-1.89732*u.deg)), 0.5e-5*u.deg)
        
    def test_true_lon(self) : 
        """tests calculation of sun's true longitude"""
        lon = self.pos.get_true_lon()
        self.assertLess(np.abs(lon - c.Angle(199.90988*u.deg)), 1.0e-5*u.deg)
        
    def test_apparent_lon(self) : 
        """tests calculation of sun's apparent longitude"""
        lon = self.pos.get_apparent_lon().wrap_at(360*u.deg)
        self.assertLess(np.abs(lon - c.Angle(199.90895*u.deg)), 1.0e-5*u.deg)
        
    def test_apparent_position(self) : 
        """tests calculation of sun's apparent position"""
        coord = self.pos.get_apparent_position()
        self.assertLess(np.abs(coord.ra.wrap_at(180*u.deg) - c.Angle(-161.61917*u.deg)), 0.5e-5*u.deg)
        self.assertLess(np.abs(coord.dec - c.Angle(-7.78507*u.deg)), 0.5e-5*u.deg)

        
class Ex15aVenus_pt1 (object) :         
    """This class exists only to give the "Uptime" class venus ephemeris data
    
    Data comes from example 15 a pg. 103
    """
    def __init__(self, time) : 
        self.disk_radius = 0 * u.deg
        
    def get_apparent_position(self) : 
        ra = 41.73129 * u.deg
        dec = 18.44092 * u.deg
        return c.SkyCoord(ra=ra, dec=dec)
        
class Ex15aVenus_pt2 (Ex15aVenus_pt1) : 
    """this class gives the uptime class the "interpolated" venus ephemeris
    
    Data comes from p. 104
    """
    def get_apparent_position(self): 
        #  transit, rise, set
        ra = [ 42.59324, 42.27648, 41.85927] * u.deg
        dec = [0, 18.64229, 18.48835] * u.deg
        return c.SkyCoord(ra=ra, dec=dec)
        
class TestUptime(unittest.TestCase) : 
    def setUp(self) : 
        self.t = t.Time("1988-03-20", scale='utc')
        self.location = c.EarthLocation.from_geodetic(-71.0833, 42.3333)
        self.up = sun.Uptime(Ex15aVenus_pt1, self.t, self.location)
        # replace calculated sidereal time with given time
        self.up.sidereal_greenwich = c.Angle( (11,50,58.10), unit=u.hour)
        
    @unittest.expectedFailure    
    def test_sidereal(self) : 
        """check that sidereal time calculation matches the "accurate ephemeris"
        
        p. 103
        Note that this is marked as an "expected failure" because the astropy
        method of calculating apparent sidereal time at greenwich does not agree
        with the "given" value to the least significant figure given. It does, 
        however, agree to within (6.5e-5 hourangle). (One least count is 2.7e-6 hourangle,
        so this is approximately a factor of 20 more than the least count.
        
        This test failure is the reason we override the computed sidereal time in
        the setup method.
        """
        sidereal = self.up.midnight_utc.sidereal_time("apparent","greenwich")
        self.assertLess(np.abs(sidereal - self.up.sidereal_greenwich), c.Angle( (0,0,0.01),unit=u.hour))
        
    def test_delta_t(self) : 
        """check the computed difference between ut and tt.
        
        From pg. 78: "...an instant given in UT is later than the instant
        in TD having the same numerical value" (UT=universal time, TD=dynamical
        time)
        """
        # self.t == UT on the UTC scale (should be later)
        dt = self.t.datetime
        
        # tt == UTC numerical time on tt scale. (should be earlier)
        tt = t.Time(dt, scale='tt')
        delta_t = (self.t - tt) # later - earlier should be positive
        
        self.assertLess(np.abs(delta_t - 56*u.s), 0.5*u.s) 
        
    def test_approximate(self) :
        """check calculation of approximate event times"""
        approx = self.up.approximate()
        
        self.assertLess(np.abs(approx[0,0] - 0.81965 * u.sday), 1*u.s)
        self.assertLess(np.abs(approx[0,1] - 0.51817 * u.sday), 1*u.s)
        self.assertLess(np.abs(approx[0,2] - 0.12113 * u.sday), 1*u.s)
        
    def test_corrections(self) : 
        """check calculation of corrections to approximate event times"""
        # force calculation of approximate event times.
        approx = self.up.approximate() 
        
        # change "body" object to object giving interpolated ephemeris
        self.up.body_class = Ex15aVenus_pt2
        cor = self.up.correction()
        
        self.assertLess(np.abs(cor[0,0] - 0.00015*u.day), 1*u.s)
        self.assertLess(np.abs(cor[0,1] - (-0.00051 * u.day)), 1*u.s)
        self.assertLess(np.abs(cor[0,2] - 0.00017 * u.day), 1*u.s)
        
    def test_approx_daylength(self) : 
        """check calculation of daylength"""
        approx_daylength = (0.12113 - 0.51817 + 1) * u.sday
        
        self.assertLess(np.abs(self.up.approximate_daylength() - approx_daylength), 1*u.s)
        
    def test_accurate_daylength(self) : 
        """check accurate calculation of daylength"""
        # force calculation of approximate event times.
        approx = self.up.approximate() 
        
        # change "body" object to object giving interpolated ephemeris
        self.up.body_class = Ex15aVenus_pt2

        accurate_daylength = ( (0.12113+0.00017) - (0.51817+(-0.00051)) + 1) * u.sday
        self.assertLess(np.abs(self.up.accurate_daylength() - accurate_daylength), 1*u.s)
        
    def test_uptime_multi_location(self) : 
        """check that uptime handles instantiation with an array of locations"""
        lats = np.arange(-80, 81, 5) * u.deg
        lons = np.zeros( (len(lats),))
        locs = c.EarthLocation.from_geodetic(lons, lats)
        
        up = sun.Uptime(sun.SunPosition, t.Time('2014-01-01'), locs)
        x = up.accurate()
        self.assertTrue(np.all(x.shape == (lats.size,3) ))
        
    def test_missoula_daylength(self) : 
        """check that daylight times match naval observatory values for missoula
        
        Data here are taken from the US Naval Observatory's webpage for 
        Missoula, MT in the year 2013.  These data reflect the length of time
        that any part of the solar disc is above the horizon.
        (http://aa.usno.navy.mil/data/docs/Dur_OneYear.php)
        """
        days = t.Time( [ '2013-01-01', '2013-03-01', '2013-05-01', '2013-07-01',
                         '2013-09-01', '2013-11-01'])
        daylengths = u.Quantity( 
             [ 8*u.hour+37*u.min, 11*u.hour+7*u.min, 14*u.hour+26*u.min,
               15*u.hour+48*u.min, 13*u.hour+19*u.min, 10*u.hour+1*u.min],
               unit = u.hour)
        missoula = c.EarthLocation.from_geodetic('-114d0m', '46d52m')
        test_daylengths = u.Quantity( np.empty(daylengths.shape), unit=u.hour)
        for i in range(daylengths.size) : 
            up = sun.Uptime(sun.SunPosition, days[i], missoula)
            test_daylengths[i] = up.accurate_daylength()
            
            
        self.assertTrue(np.all(np.abs(test_daylengths-daylengths)<3.5*u.min))
        
    def test_prudhoe_bay_daylength(self) : 
        """check that daylight times match naval observatory values for prudhoe bay, AK
        
        Data here are taken from the US Naval Observatory's webpage for 
        Prudhoe Bay, AK in the year 2013.  These data reflect the length of time
        that any part of the solar disc is above the horizon. This test includes
        time where the sun is up for the entire day as well as time where it is
        down for the entire day.
        (http://aa.usno.navy.mil/data/docs/Dur_OneYear.php)
        """
        days = t.Time( [ '2013-01-01', '2013-03-01', '2013-05-01', '2013-07-01',
                         '2013-09-01', '2013-11-01'])
        daylengths = u.Quantity(
            [ 0*u.min, 9*u.hour+35*u.min, 19*u.hour+11*u.min, 24*u.hour, 
              15*u.hour+25*u.min, 6*u.hour+14*u.min], unit=u.hour)
        prudhoe = c.EarthLocation.from_geodetic('-148d22m','70d15m')
        test_daylengths = u.Quantity( np.empty(daylengths.shape), unit=u.hour)
        for i in range(daylengths.size) : 
            up = sun.Uptime(sun.SunPosition, days[i], prudhoe)
            test_daylengths[i] = up.accurate_daylength()
            
        self.assertTrue(np.all(np.abs(test_daylengths-daylengths)<9*u.min))
        
    def test_tiksi_siberia_daylength(self) : 
        """check that daylight times match naval observatory values for Tiksi, Siberia
    
        Data here are taken from the US Naval Observatory's webpage for 
        Tiksi, Siberia in the year 2013.  These data reflect the length of time
        that any part of the solar disc is above the horizon. This test includes
        time where the sun is up for the entire day as well as time where it is
        down for the entire day, and covers the case of "east-of-the-prime-meridian"
        (http://aa.usno.navy.mil/data/docs/Dur_OneYear.php)
        """
        days = t.Time( [ '2013-01-01', '2013-03-01', '2013-05-01', '2013-07-01',
                         '2013-09-01', '2013-11-01'])
        daylengths = u.Quantity(
            [ 0*u.min, 9*u.hour+20*u.min, 20*u.hour+3*u.min, 24*u.hour, 
              15*u.hour+46*u.min, 5*u.hour+45*u.min], unit=u.hour)
        tiksi = c.EarthLocation.from_geodetic('128d52m','71d39m')
        test_daylengths = u.Quantity( np.empty(daylengths.shape), unit=u.hour)
        for i in range(daylengths.size) : 
            up = sun.Uptime(sun.SunPosition, days[i], tiksi)
            test_daylengths[i] = up.accurate_daylength()
            
        self.assertTrue(np.all(np.abs(test_daylengths-daylengths)<9*u.min))
    
    
