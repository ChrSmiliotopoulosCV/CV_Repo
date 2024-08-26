CREATE TABLE IF NOT EXISTS arrests (
    arrest varchar(64),
    age varchar(6),
    sex varchar(2),
    race varchar(2),
    arrest_date varchar(10),
    arrest_time varchar(10),
    arrest_location varchar(64),
    incident_offense varchar(64),
    incident_location varchar(64),
    charge varchar(16),
    charge_description varchar(64),
    district varchar(32),
    post varchar(5),
    neighborhood varchar(64),
    location_1 varchar(64)
);