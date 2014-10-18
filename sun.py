"""
Simple, approximate calculations for solar position and characteristics.

This module implements calculations pertaining to the calculation of 
solar coordinates given in [1]. These calculations predict the geometric 
position of the sun to 0.01 degrees by assuming a purely elliptical Earth
orbit. 

[1] Meeus, Jean. Astronomical Algorithms. 2nd edition. Richmond, Va: 
    Willmann-Bell, 1998.
"""
import astropy.units as u
import astropy.time as time
import astropy.coordinates as c
import datetime as dt
import numpy as np

julian_century = u.def_unit("julian_century", 36525*u.day)
j2000_epoch = time.Time(val='2000:001:12:00', scale='tt', format='yday')


def jc(d) : 
    """Converts a standard date to a time delta from J2000 epoch
    
    Most, if not all of the equations in this file are empirical fits 
    using delta time from an epoch."""
    return (d-j2000_epoch).to(julian_century)

def day(d, scale) : 
    """converts a full timestamp to a date
    """
    date = d.datetime.date()
    d1 = dt.datetime(date.year,date.month,date.day)
    return time.Time(d1,scale=scale)

def geometric_mean_longitude(t) : 
    """geometric mean longitude of the sun referred to mean equinox of the date
    
    Equation 25.2, p 163.
    
    Parameters
    ----------
    t : array : Julian centuries (36525 days)
    
    Returns
    -------
    L0 : array: deg
    """
    return 280.46646*u.deg + (36000.76983*u.deg/julian_century)*t + (0.0003032*u.deg/(julian_century**2))*(t**2)
    
def mean_anomaly(t) : 
    """Mean anomaly of the sun (earth)
    
    Equation 25.3, p. 163, different from equation on p. 144
    
    Parameters
    ----------
    t : array : Julian centuries
    
    Returns
    -------
    M : array : deg
    """
    return (357.52911*u.deg + 
            (35999.05029*u.deg/julian_century)*t +
            (0.0001537*u.deg/(julian_century**2))*(t**2))
            
def equation_of_center(t, m) :
    """Sun equation of center
     
    Unnumbered equation, top of page 164.
      
    Parameters
    ----------
    t : array : julian centuries
    
    m : array : deg
        solar/earth mean anomaly
        
    Returns
    -------
    c : array : deg
    """
    return ( (1.914602*u.deg - (0.004817*u.deg/julian_century)*t - (0.000014*u.deg/(julian_century**2))*t**2)*np.sin(m)
             +(0.019993*u.deg - (0.000101*u.deg/julian_century)*t)*np.sin(2*m)
             +(0.000289*u.deg*np.sin(3*m)))

def mean_ecliptic_obliquity(t) : 
    """calculates the mean obliquity of the ecliptic at the specified time
    
    Equation 22.2, pg. 147. The accuracy of this equation is estimated to 
    be 0.01 arcseconds for 1000 years to either side of the epoch in 2000 A.D.
    
    Parameters
    ----------
    t : array : julian centuries
    
    Returns
    -------
    epsilon0  : array : deg
    """
    return ( c.Angle( (23,26,21.448), unit=u.deg) 
            - (46.8150 * u.arcsec / julian_century)*t
            - (0.00059 * u.arcsec / (julian_century**2)*(t**2))
            + (0.001813 * u.arcsec / (julian_century**3)*(t**3)))


def lon_moon_ascending_node(t) :
    """Longitude of the ascending node of the moon's mean orbit on the ecliptic
    
    Unnumbered equation, pg. 144.
    """
    return ( 125.04452*u.deg - (1934.136261*u.deg/julian_century)*t +
              (0.0020708 * u.deg/(julian_century**2))*(t**2) + 
              (t**3)/(450000 * (julian_century**3)/u.deg) )


def body_altitude(dec, obs_lat, hour_angle) : 
    """Calculates the altitude (elevation) of a body given the sky coords and hour angle
    
    Equation 13.6, p 93
    
    """
    
    sin_alt = np.sin(obs_lat)*np.sin(dec) + np.cos(obs_lat)*np.cos(dec)*np.cos(hour_angle)
    return np.arcsin(sin_alt)
    
def v_body_altitude(dec, obs_lat, hour_angle) : 
    """vectorized version of the body altitude calculation
    
    the size of declination and hour angle arrays are correlated with the number
    of observation times. The number of observation lats is independent. These 
    should form a 2D array of shape: (obs_lat.size,dec.size) This function 
    loops over the body_altitude function once for each observation latitude.
    """
    if obs_lat.size > 1 : 
        sin_alt = u.Quantity(np.empty(obs_lat.size, dec.size), unit=u.dimensionless_unscaled)
        for i in range(obs_lat.size) : 
            sin_alt[i,:] = body_altitude(dec[i,:], obs_lat[i], hour_angle[i,:])
    else : 
        sin_alt = body_altitude(dec,obs_lat, hour_angle)
    return sin_alt

    
def fix_day(time) : 
    """bounds the fractional day to [0,1]"""
    return time % (1*time.unit)
    

class Ecliptic(object) : 
    def __init__(self, time) : 
        self.time = time 
        self.jc_time = jc(time)
        self._mean_obliquity = None
        self._moon_asc_lon = None
        self._nutation_obliquity = None
        self._true_obliquity = None
        
    def get_mean_obliquity(self) : 
        if self._mean_obliquity == None : 
            self._mean_obliquity = mean_ecliptic_obliquity(self.jc_time)
        return self._mean_obliquity
        
    def get_moon_asc_lon(self) : 
        if self._moon_asc_lon == None : 
            self._moon_asc_lon = lon_moon_ascending_node(self.jc_time)
        return self._moon_asc_lon
        
    def get_nutation_obliquity(self) :
        """calculate the nutation in obliquity
        
        Equation 25.8, p 165.
        This is a very course approximation, which is essentially the first term
        in a somewhat finer approximation toward the bottom
        of page 144. 
        """
        if self._nutation_obliquity == None : 
            omega = self.get_moon_asc_lon()
            self._nutation_obliquity = 9.2*u.arcsec * np.cos(omega)
        return self._nutation_obliquity
        
    def get_true_obliquity(self) : 
        """calculate the true obliquity
        
        This is the mean obliquity + the nutation in obliquity.
        """
        if self._true_obliquity == None :
            delta_eps = self.get_nutation_obliquity()
            eps0 = self.get_mean_obliquity()
            self._true_obliquity = eps0 + delta_eps
        return self._true_obliquity
                  
class SunPosition(object) : 
    """Orchestrates the various positional calculations in order to minimize redundancy."""
    def __init__(self, time) : 
        self.time = time
        self.jc_time = jc(time)
        self.disk_radius = 16*u.arcmin
        self._mean_longitude = None
        self._mean_anomaly = None
        self._eoc = None
        self._true_longitude = None
        self._true_anomaly = None
        self._apparent_longitude = None
        self._true_pos = None
        self._apparent_pos = None
        self._ecliptic = None
    
    def get_mean_lon(self) : 
        if self._mean_longitude == None : 
            self._mean_longitude = geometric_mean_longitude(self.jc_time)
        return self._mean_longitude
    
    def get_mean_anomaly(self) : 
        if self._mean_anomaly == None : 
            self._mean_anomaly = mean_anomaly(self.jc_time)
        return self._mean_anomaly
        
    def get_eoc(self) : 
        if self._eoc == None : 
            m = self.get_mean_anomaly()
            self._eoc = equation_of_center(self.jc_time, m)
        return self._eoc
    
    def get_true_lon(self):  
        """calculate true longitude
        
        Unnumbered equation, p. 164
        """
        if self._true_longitude == None : 
            l0 = self.get_mean_lon()
            eoc = self.get_eoc()
            self._true_longitude = c.Angle(l0 + eoc).wrap_at(360*u.deg)
        return self._true_longitude
        
    def get_true_anomaly(self) : 
        """calculate true anomaly
        
        Unnumbered equation, p. 164
        """
        if self._true_anomaly == None : 
            m = self.get_mean_anomaly()
            eoc = self.get_eoc()
            self._true_anomaly = m + eoc
        return self._true_anomaly
        
    def get_apparent_lon(self) : 
        """calculate the apparent solar longitude
        
        Unnumbered equation, p. 164
        """
        if self._apparent_longitude == None : 
            omega = 125.04*u.deg - (1934.136*u.deg/julian_century)*self.jc_time
            lon = self.get_true_lon()
            self._apparent_longitude = c.Angle(lon - 0.00569*u.deg - (0.00478*u.deg)*np.sin(omega)).wrap_at(360*u.deg)
        return self._apparent_longitude
        
    def get_ecliptic(self) : 
        if self._ecliptic == None : 
            self._ecliptic = Ecliptic(self.time)
        return self._ecliptic
        
    def _get_position(self, epsilon, lon) : 
        """calculates the sun position given the longitude
        
        Equations 25.6 and 25.7 on p. 165. If true position is desired, 
        pass the true longitude, if apparent position, pass apparent longitude.
        
        Parameters
        ----------
        epsilon : array : deg
            mean or true obliquity of the elliptic
        lon : array : deg
            longitude, either apparent or true
        
        Returns
        -------
        pos : array : deg
            SkyCoord object populated with the ra and dec of the sun's position
        """
        sinlon = np.sin(lon)
        ra = np.arctan2(np.cos(epsilon)*sinlon, np.cos(lon))
        dec = np.arcsin(np.sin(epsilon)*sinlon)
        return c.SkyCoord(ra=ra, dec=dec, obstime=self.time)
        
    def get_true_position(self) : 
        if self._true_pos == None : 
            epsilon = self.get_ecliptic().get_mean_obliquity()
            self._true_pos = self._get_position(epsilon, self.get_true_lon())
        return self._true_pos
        
    def get_apparent_position(self) : 
        if self._apparent_pos == None : 
            epsilon = self.get_ecliptic().get_true_obliquity()
            self._apparent_pos = self._get_position(epsilon, self.get_apparent_lon())
        return self._apparent_pos
        
class Uptime(object) : 
    def __init__(self, body_class, time, obs_location, refraction=(34*u.arcmin)) : 
        self.obs_location = obs_location
        self.refraction = refraction
        self.midnight_utc = day(time, 'utc')
        self.midnight_tt  = day(time, 'tt')
        self.body_class = body_class
        self.body = body_class(self.midnight_utc)
        self.h0 = - (refraction + self.body.disk_radius)
        self.sidereal_greenwich = self.midnight_utc.sidereal_time('apparent','greenwich')
        self._approx_m = None
        self._correction_m = None
        
    def approximate(self) : 
        """performs the approximate calculation of sunrise/sunset times
        
        Equation 15.1 and 15.2, p. 102
        These calculations are good to approximately +-0.01 days (+-14.4 minutes)
        """
        if self._approx_m != None : 
            return self._approx_m
            
            
        pos = self.body.get_apparent_position()
        dec = pos.dec
        ra  = pos.ra
        lat = self.obs_location.latitude
        lon = self.obs_location.longitude
        
        # eq 15.1
        cos_H0 = ((np.sin(self.h0) - (np.sin(lat)*np.sin(dec)))/
                  (np.cos(lat)*np.cos(dec)))
        np.clip(cos_H0, -1, 1)
        H0 = np.arccos(cos_H0)
        
        # eq 15.2
        # transit time, fractional day
        # modified from eq. 15.2 because 15.2 expects longitude to be 
        # positive "west" instead of positive "east".
        m0 = (ra - lon - self.sidereal_greenwich)/(360*u.deg/u.sday)
        # delta time, transit to sunrise/set
        delta_t = H0/(360*u.deg/u.sday)
        m1 = m0 - delta_t
        m2 = m0 + delta_t
        self._approx_m = u.Quantity( np.empty((m0.size, 3), dtype=m0.dtype), unit=m0.unit)
        print fix_day(m0)
        print m0.size
        print self._approx_m.shape
        self._approx_m[:,0] = fix_day(m0)
        self._approx_m[:,1] = fix_day(m1)
        self._approx_m[:,2] = fix_day(m2)
        return self._approx_m
    
    def _correct_rise_set(self, h, dec, lat, hour_angle):
        """implements correction to rise/set time offsets
        
        Unnumbered equation, p. 103
        """
        delta_m = (h-self.h0) / ((360*u.deg/u.day)*np.cos(dec)*np.cos(lat)*np.sin(hour_angle))
        return delta_m
        
    
    def correction(self) :
        """Calculates a correction to transit/rise/set times
        
        Implements unnumbered equations on p. 103, defining a correction
        to m. These should be relatively small, on the order of +-0.01 days
        (+-14 minutes)
        """ 
        if self._correction_m != None : 
            return self._correction_m
            
        # calculate sidereal time at each event
        # scale m such that it represents the same fraction of a solar day that 
        # it represented in a sidereal day. Because a solar day is longer,
        # the fraction of the day will also be longer.
        # also change units.
        #m_solar_day = (self.approximate()*1.0027379093604878) * (u.day/u.sday)
        m_sidereal = self.approximate()
        m_solar_day = c.Angle(m_sidereal*(360.985647 * u.deg/u.sday))
        print self.midnight_utc
        print repr(m_solar_day/(360*u.deg/u.sday))
        
        # sidereal times at greenwich of the events (transit/rise/set)
        # have to convert to hour, or else it won't add to an hourangle
        #sidereal_t = self.sidereal_greenwich + c.Angle(m_solar_day.to(u.hour))
        sidereal_t = self.sidereal_greenwich + m_solar_day
        sidereal_t.wrap_at(360*u.deg, inplace=True)
        
        # times of the events
        #t_events = self.midnight_utc + (m_solar_day/(360*u.deg/u.sday)).to(u.sday)
        t_events = self.midnight_utc + m_sidereal.to(u.hour)
        
        # calculate apparent positions at the time of the events
        body = self.body_class(t_events.tt)
        pos = body.get_apparent_position()
        
        # calculate local hour angle of the body
        # changed sign on longitude because formula in book expects
        # longitudes to be numbered positive westward.
        H = sidereal_t + self.obs_location.longitude - pos.ra
        
        # calculate altitude of the body for rise/set events
        alt = v_body_altitude(pos[:,1:].dec,self.obs_location.latitude, H[:,1:]) 
        
        # calculate the corrections
        dm0 = - (H[:,0]/(360*u.deg/u.day))
        dm1 = self._correct_rise_set(alt[:,0], pos[:,1].dec,self.obs_location.latitude,H[:,1])
        dm2 = self._correct_rise_set(alt[:,1], pos[:,2].dec,self.obs_location.latitude,H[:,2])
        
        self._correction_m = u.Quantity(np.array( m_sidereal.shape, dtype=m_sidereal.dtype), unit=u.day)
        self._correction_m[:,0] = dm0
        self._correction_m[:,1] = dm1
        self._correction_m[:,2] = dm2 
        return self._correction_m
        
    def accurate(self) : 
        """produces a more accurate set of times than approximate"""
        return self.approximate() + self.correction()
        
    def approximate_times(self): 
        """returns actual timestamps for the events instead of offsets within the day"""
        return self.midnight_utc + self.approximate()
        
    def accurate_times(self) : 
        """returns actual timestamps for the events instead of offsets within the day"""
        return self.midnight_utc + self.accurate()
        
    def approximate_daylength(self): 
        times = self.approximate()
        
        daylength = times[:,2] - times[:,1]
        # note: np.choose strips off units.
        # we're messing with the daylength because if the rise or set time was on
        # the previous/next UTC day, then 1 day was added/subtracted. The effect
        # will be that the sun sets before it rises, and hence a blind subtraction
        # yields the negative of the nightlength. Add one to get daylength
        daylength = np.choose(daylength<0*u.sday, [daylength, daylength+1*u.sday])
        return daylength * u.sday
        
    def accurate_daylength(self) : 
        times = self.accurate()
        daylength = times[:,2] - times[:,1]
        daylength = np.choose(daylength<0*u.sday, [daylength, daylength+1*u.sday])
        return daylength * u.sday