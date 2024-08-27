package db.core.infra;

import db.core.model.AnonymizedRecord;
import java.util.Collections;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.Optional;

@Component
public final class DbUtils {

    private final Logger logger = LoggerFactory.getLogger(DbUtils.class);

    private final SqlQueriesConfig sqlQueriesConfig;
    private final JdbcTemplate jdbcTemplate;
    private final AnonymizedRecordMapper anonymizedRecordMapper;

    @Autowired
    public DbUtils(
        SqlQueriesConfig sqlQueriesConfig,
        JdbcTemplate jdbcTemplate,
        AnonymizedRecordMapper anonymizedRecordMapper
    ) {
        this.sqlQueriesConfig = sqlQueriesConfig;
        this.jdbcTemplate = jdbcTemplate;
        this.anonymizedRecordMapper = anonymizedRecordMapper;
    }

    public int getNumOfRecords() {
        var countRecords = jdbcTemplate.queryForObject(sqlQueriesConfig.getCountRecords(), Integer.class);
        return Optional.ofNullable(countRecords)
                       .orElse(0);
    }

    public List<AnonymizedRecord> getAnonymizedRecords() {
        return jdbcTemplate.query(sqlQueriesConfig.getGetAnonymizedRecords(), anonymizedRecordMapper);
    }
}
