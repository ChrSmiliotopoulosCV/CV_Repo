package db.core.business;

import db.core.infra.DbUtils;
import db.core.model.AnonymizedRecord;
import java.time.LocalDate;
import java.time.format.DateTimeFormatter;
import java.util.List;
import java.util.function.Consumer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

/**
 * this class creates the generalization domains for the categorical data
 * we provide
 */
@Component
public final class GeneralizationTaxonomyBuilder {

    private final Logger logger = LoggerFactory.getLogger(GeneralizationTaxonomyBuilder.class);

    private final GeneralizationStatus generalizationStatus;

    private final DateTimeFormatter dateTimeFormatter = DateTimeFormatter.ofPattern("MM/dd/yyyy");

    @Autowired
    public GeneralizationTaxonomyBuilder(DbUtils dbUtils) {
        generalizationStatus = new GeneralizationStatus();
    }

    // 0-10, 11-20, 21-30, 31-40, 41-50, 51-60, 61-70, 71-80, 81-90, 91-100, 100+
    private AnonymizedRecord ageGeneralizeL1(AnonymizedRecord record) {
        int ageNum = Integer.parseInt(record.getAge());
        var result = "100+";

        if (ageNum <= 10.0) {
            result = "0-10";
        } else if (ageNum <= 20.0) {
            result = "11-20";
        } else if (ageNum <= 30.0) {
            result = "21-30";
        } else if (ageNum <= 40.0) {
            result = "31-40";
        } else if (ageNum <= 50.0) {
            result = "41-50";
        } else if (ageNum <= 60.0) {
            result = "51-60";
        } else if (ageNum <= 70.0) {
            result = "61-70";
        } else if (ageNum <= 80.0) {
            result = "71-80";
        } else if (ageNum <= 90.0) {
            result = "81-90";
        } else if (ageNum <= 100.0) {
            result = "91-100";
        }
        
        logger.debug("Age: {} -> L1 category: {}", ageNum, result);

        record.setAge(result);
        return record;
    }

    // 0-30, 31-60, 61-90, 90+
    private AnonymizedRecord ageGeneralizeL2(AnonymizedRecord record) {
        var ageNum = Integer.parseInt(record.getAge().split("-")[0]);
        var result = "90+";
        
        if (ageNum <= 30.0) {
            result = "0-30";
        } else if (ageNum <= 60.0) {
            result = "31-60";
        } else if (ageNum <= 90.0) {
            result = "61-90";
        }
        
        logger.debug("Age: {} -> L2 category: {}", ageNum, result);

        record.setAge(result);
        return record;
    }

    private AnonymizedRecord ageGeneralizeL3(AnonymizedRecord record) {
        logger.debug("Suppressing age {}", record.getAge());
        record.setAge("*");
        return record;
    }

    private AnonymizedRecord sexGeneralizeL1(AnonymizedRecord record) {
        logger.debug("Suppressing gender {}", record.getSex());
        record.setSex("*");
        return record;
    }

    private AnonymizedRecord districtGeneralizeL1(AnonymizedRecord record) {
        record.setDistrict("*"); // this goes to Baltimore, so essentially suppressed
        return record;
    }

    private AnonymizedRecord raceGeneralizeL1(AnonymizedRecord record) {
        logger.debug("Race: {} -> North-American", record.getRace());
        record.setRace("North-American");
        return record;
    }

    private AnonymizedRecord raceGeneralizeL2(AnonymizedRecord record) {
        logger.debug("Suppressing race: {}", record.getRace());
        record.setRace("*");
        return record;
    }

    private AnonymizedRecord arrestDateGeneralizeL1(AnonymizedRecord record) {
        logger.debug("record: {}", record);
        logger.debug("Arrest Date: {} -> same month", record.getArrestDate());
        LocalDate date = LocalDate.parse(record.getArrestDate(), dateTimeFormatter);
        record.setArrestDate(LocalDate.of(date.getYear(), date.getMonthValue(), 1).format(dateTimeFormatter));
        return record;
    }

    private AnonymizedRecord arrestDateGeneralizeL2(AnonymizedRecord record) {
        logger.debug("Arrest Date: {} -> same year", record.getArrestDate());
        logger.debug("{}", record);
        LocalDate date = LocalDate.parse(record.getArrestDate(), dateTimeFormatter);
        record.setArrestDate(LocalDate.of(date.getYear(), 1, 1).format(dateTimeFormatter));
        return record;
    }

    private AnonymizedRecord arrestDateGeneralizeL3(AnonymizedRecord record) {
        logger.debug("Suppressing Arrest Date");
        record.setArrestDate("*");
        return record;
    }

    public GeneralizationStatus increaseGeneralizationDomain(
            List<AnonymizedRecord> anonymizedRecords
    ) {

        if (generalizationStatus.getAgeCurrentDomain() == 0) {
            generalizationStatus.increaseAgeDomain();
            generalizeRecords(anonymizedRecords, this::ageGeneralizeL1);
        } else if (generalizationStatus.getDistrictCurrentDomain() == 0) {
            generalizationStatus.increaseDistrictDomain();
            generalizeRecords(anonymizedRecords, this::districtGeneralizeL1);
        } else if (generalizationStatus.getSexCurrentDomain() == 0) {
            generalizationStatus.increaseSexDomain();
            generalizeRecords(anonymizedRecords, this::sexGeneralizeL1);
        } else if (generalizationStatus.getRaceCurrentDomain() == 0) {
            generalizationStatus.increaseRaceDomain();
            generalizeRecords(anonymizedRecords, this::raceGeneralizeL1);
        } else if (generalizationStatus.getArrestCurrentDomain() == 0) {
            generalizationStatus.increaseArrestDateDomain();
            generalizeRecords(anonymizedRecords, this::arrestDateGeneralizeL1);
        } else if (generalizationStatus.getAgeCurrentDomain() == 1) {
            generalizationStatus.increaseAgeDomain();
            generalizeRecords(anonymizedRecords, this::ageGeneralizeL2);
        } else if (generalizationStatus.getRaceCurrentDomain() == 1) {
            generalizationStatus.increaseRaceDomain();
            generalizeRecords(anonymizedRecords, this::raceGeneralizeL2);
        } else if (generalizationStatus.getArrestCurrentDomain() == 1) {
            generalizationStatus.increaseArrestDateDomain();
            generalizeRecords(anonymizedRecords, this::arrestDateGeneralizeL2);
        } else if (generalizationStatus.getAgeCurrentDomain() == 2) {
            generalizationStatus.increaseAgeDomain();
            generalizeRecords(anonymizedRecords, this::ageGeneralizeL3);
        } else if (generalizationStatus.getArrestCurrentDomain() == 2) {
            generalizationStatus.increaseArrestDateDomain();
            generalizeRecords(anonymizedRecords, this::arrestDateGeneralizeL3);
        }

        return generalizationStatus;
    }
    
    private void generalizeRecords(List<AnonymizedRecord> records, Consumer<AnonymizedRecord> action) {
        for(AnonymizedRecord record: records) {
            action.accept(record);
        }
    }

    public boolean isNotFullySuppressed() {
        return generalizationStatus.isNotFullySuppressed();
    }
    
    public double getInformationLoss() {
        return generalizationStatus.getInformationLoss();
    }

    public String getCurrentGeneralizationLevels() {
        return generalizationStatus.getCurrentGeneralizationLevels();
    }

    public String describeGeneralizationDomains() {
        return generalizationStatus.describeGeneralizationDomains();
    }

    public void reset() {
        generalizationStatus.reset();
    }

    public static class GeneralizationStatus {

        private final Logger logger = LoggerFactory.getLogger(GeneralizationStatus.class);

        private final List<String> ageDomains;
        private int ageCurrentDomain;
        private final List<String> sexDomains;
        private int sexCurrentDomain;
        private final List<String> raceDomains;
        private int raceCurrentDomain;
        private final List<String> districtDomains;
        private int districtCurrentDomain;
        private final List<String> arrestDateDomains;
        private int arrestCurrentDomain;

        public int getAgeCurrentDomain() {
            return ageCurrentDomain;
        }

        public void increaseAgeDomain() {
            if (ageCurrentDomain < 3) {
                ageCurrentDomain++;
            }
        }

        public int getSexCurrentDomain() {
            return sexCurrentDomain;
        }

        public void increaseSexDomain() {
            if (sexCurrentDomain < 1) {
                sexCurrentDomain++;
            }
        }

        public int getRaceCurrentDomain() {
            return raceCurrentDomain;
        }

        public void increaseRaceDomain() {
            if (raceCurrentDomain < 2) {
                raceCurrentDomain++;
            }
        }

        public int getDistrictCurrentDomain() {
            return districtCurrentDomain;
        }

        public void increaseDistrictDomain() {
            if (districtCurrentDomain < 1) {
                districtCurrentDomain++;
            }
        }

        public int getArrestCurrentDomain() {
            return arrestCurrentDomain;
        }

        public void increaseArrestDateDomain() {
            if (arrestCurrentDomain < 3) {
                arrestCurrentDomain++;
            }
        }

        public GeneralizationStatus() {
            ageDomains = List.of("L0", "L1", "L2", "*"); // distinct, every 10 years, every 30 years, any
            ageCurrentDomain = 0;

            sexDomains = List.of("L0", "*"); // distinct, any
            sexCurrentDomain = 0;

            raceDomains = List.of("L0", "L1", "*"); // distinct, North-American, any
            raceCurrentDomain = 0;

            districtDomains = List.of("L0", "*"); // district, any (baltimore)
            districtCurrentDomain = 0;

            arrestDateDomains = List.of("L0", "L1", "L2", "*"); // day/month/year, month/year, year, any
            arrestCurrentDomain = 0;
        }

        public void reset() {
            ageCurrentDomain = 0;
            sexCurrentDomain = 0;
            raceCurrentDomain = 0;
            districtCurrentDomain = 0;
            arrestCurrentDomain = 0;
        }

        public boolean isNotFullySuppressed() {
            return ageCurrentDomain < 3 || sexCurrentDomain < 1 || raceCurrentDomain < 2 || districtCurrentDomain < 1 || arrestCurrentDomain < 3;
        }

        public double getInformationLoss() {
            int currentDomains = ageCurrentDomain + sexCurrentDomain + raceCurrentDomain + districtCurrentDomain + arrestCurrentDomain;
            int domainDepths = (ageDomains.size()-1) + (sexDomains.size()-1) + (raceDomains.size()-1) + (districtDomains.size()-1) + (arrestDateDomains.size()-1);
            logger.debug("age: {} sex: {} race: {} district: {} arrest date: {}, totalDomains: {}, totalDepths: {}, infoloss: {}",
                    ageCurrentDomain, sexCurrentDomain,
                    raceCurrentDomain, districtCurrentDomain, arrestCurrentDomain,
                    currentDomains,
                    domainDepths,
                    (double) currentDomains/(double)domainDepths
            );
            return (double) currentDomains/(double)domainDepths;
        }

        public String getCurrentGeneralizationLevels() {
            return "age -> " + ageDomains.get(ageCurrentDomain) + "\n" +
                    "district -> " + districtDomains.get(districtCurrentDomain) + "\n" +
                    "race -> " + raceDomains.get(raceCurrentDomain) + "\n" +
                    "sex -> " + sexDomains.get(sexCurrentDomain) + "\n" +
                    "arrest date -> " + arrestDateDomains.get(arrestCurrentDomain) + "\n";
        }

        public String describeGeneralizationDomains() {
            return "age:\n" +
                    "L0: 0-10, 11-20, 21-30, 31-40, 41-50, 51-60, 61-70, 71-80, 81-90, 91-100, 100+\n" +
                    "L1: 0-30, 31-60, 61-90, 90+\n" +
                    "L2: *\n" +
                    "\ndistrict: \n" +
                    "L0: (all districts in the dataset)\n" +
                    "L1: *\n" +
                    "\nrace:\n" +
                    "L0: Asian, Black, Hispanic, Native, Other, White\n" +
                    "L1: North-American\n" +
                    "L2: *\n" +
                    "\nsex:\n" +
                    "L0: M, F\n" +
                    "L1 *\n" +
                    "\narrest date:\n" +
                    "L0: month/day/year\n" +
                    "L1: month/year \n" +
                    "L2: year\n" +
                    "L3: *\n";
        }
    }
}
