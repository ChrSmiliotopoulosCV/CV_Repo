package db.core.infra;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Configuration;

@Configuration
public class SqlQueriesConfig {

    private final String countRecords;
    private final String getAnonymizedRecords;

    @Autowired
    public SqlQueriesConfig(
            @Value("${sql.queries.countRecords}") String countRecords,
            @Value("${sql.queries.getAnonymizedRecords}") String getAnonymizedRecords
    ) {
        this.countRecords = countRecords;
        this.getAnonymizedRecords = getAnonymizedRecords;
    }

    public String getCountRecords() {
        return countRecords;
    }

    public String getGetAnonymizedRecords() {
        return getAnonymizedRecords;
    }
}

