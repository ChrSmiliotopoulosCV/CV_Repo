package db.core.business;

import db.core.infra.DbUtils;
import db.core.model.AnonymizedRecord;
import java.text.DecimalFormat;
import java.util.List;
import java.util.stream.Collectors;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.context.event.ApplicationReadyEvent;
import org.springframework.context.event.EventListener;
import org.springframework.stereotype.Component;

import static java.util.stream.Collectors.groupingBy;
import static java.util.stream.Collectors.mapping;
import static java.util.stream.Collectors.toList;

@Component
public class ApplicationAutoRunner {

    private final Logger logger = LoggerFactory.getLogger(ApplicationAutoRunner.class);

    private final DbUtils dbUtils;
    private final GeneralizationTaxonomyBuilder generalizationTaxonomyBuilder;

    @Autowired
    public ApplicationAutoRunner(
            DbUtils dbUtils,
            GeneralizationTaxonomyBuilder generalizationTaxonomyBuilder
    ) {
        this.dbUtils = dbUtils;
        this.generalizationTaxonomyBuilder = generalizationTaxonomyBuilder;
    }

    @EventListener
    public void run(ApplicationReadyEvent event) {

        String generalizationDomainsDescription = generalizationTaxonomyBuilder.describeGeneralizationDomains();
        logger.warn("\nDescription of the generalization domains used in this program:\n{}", generalizationDomainsDescription);

        var totalRecords = dbUtils.getNumOfRecords();
        logger.warn("Loaded {} anonymized records \n", totalRecords);

        var anonymizedRecords = dbUtils.getAnonymizedRecords();

        var cleanedAnonymizedRecords = anonymizedRecords.stream()
                                                        .filter(this::hasAllValues)
                                                        .collect(Collectors.toList());

        logger.warn("Loaded {} cleaned records\n", cleanedAnonymizedRecords.size());

        var kAnonymizer = new KAnonymity(cleanedAnonymizedRecords, generalizationTaxonomyBuilder);

        for (int i=1; i<50; i++) {
            kAnonymizer.check(i);
        }

        var cleanedAnonymizedRecords1 = dbUtils.getAnonymizedRecords()
                                               .stream()
                                               .filter(this::hasAllValues)
                                               .collect(Collectors.toList());

        var ldiversity = new LDiversity();
        var kAnonymizer1 = new KAnonymity(cleanedAnonymizedRecords1, generalizationTaxonomyBuilder, ldiversity);
        for (int i=1; i<50; i++) {
            kAnonymizer1.check(i);
        }

    }

    private boolean hasAllValues(AnonymizedRecord record) {
        if (record.getSex() == null || record.getSex().isEmpty()) {
            return false;
        }

        if (record.getAge() == null || record.getAge().isEmpty()) {
            return false;
        }

        if (record.getDistrict() == null || record.getDistrict().isEmpty()) {
            return false;
        }

        if (record.getRace() == null || record.getRace().isEmpty()) {
            return false;
        }

        if (record.getArrestDate() == null || record.getArrestDate().isEmpty()) {
            return false;
        }

        return true;
    }
}
