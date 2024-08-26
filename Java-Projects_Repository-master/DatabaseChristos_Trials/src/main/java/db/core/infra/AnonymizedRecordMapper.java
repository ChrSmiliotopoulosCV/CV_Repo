package db.core.infra;

import db.core.model.AnonymizedRecord;
import java.time.LocalDate;
import java.time.format.DateTimeFormatter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.jdbc.core.RowMapper;
import org.springframework.stereotype.Component;

import java.sql.ResultSet;
import java.sql.SQLException;

@Component
public final class AnonymizedRecordMapper implements RowMapper<AnonymizedRecord> {

    private final Logger logger = LoggerFactory.getLogger(AnonymizedRecordMapper.class);

    @Override
    public AnonymizedRecord mapRow(final ResultSet rs, final int rowNum) throws SQLException {

        var arrest = rs.getString("arrest");
        var race = rs.getString("race");
        var district = rs.getString("district");
        var sex = rs.getString("sex");
        var age = rs.getString("age");
        var arrestDate = rs.getString("arrest_date");
        var chargeDescription = rs.getString("charge_description");

        return new AnonymizedRecord(arrest, race, district, sex, age, arrestDate, chargeDescription);
    }
}
