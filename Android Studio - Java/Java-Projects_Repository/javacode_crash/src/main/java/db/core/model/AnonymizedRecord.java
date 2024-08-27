package db.core.model;

import java.util.Objects;

public final class AnonymizedRecord {

    /*
     * Μοναδικά αναγνωριστικά (uniqueidentifiers) : νούμερο πλειάδας
     * Ψευδο-αναγνωριστικά (Quasi-Identifiers -QI): race, city, sex, age, ArrestDate
     * Τα ευαίσθητα γνωρίσματα(Sensitive Attributes -SA): ChargeDescription
     */
    private String arrest;
    private String race;
    private String district;
    private String sex;
    private String age;
    private String arrestDate;
    private String chargeDescription;

    public String getArrest() {
        return arrest;
    }

    public void setArrest(String arrest) {
        this.arrest = arrest;
    }

    public String getRace() {
        return race;
    }

    public void setRace(String race) {
        this.race = race;
    }

    public String getDistrict() {
        return district;
    }

    public void setDistrict(String district) {
        this.district = district;
    }

    public String getSex() {
        return sex;
    }

    public void setSex(String sex) {
        this.sex = sex;
    }

    public String getAge() {
        return age;
    }

    public void setAge(String age) {
        this.age = age;
    }

    public String getArrestDate() {
        return arrestDate;
    }

    public void setArrestDate(String arrestDate) {
        this.arrestDate = arrestDate;
    }

    public String getChargeDescription() {
        return chargeDescription;
    }

    public void setChargeDescription(String chargeDescription) {
        this.chargeDescription = chargeDescription;
    }

    public AnonymizedRecord() {
    }

    public AnonymizedRecord(String arrest, String race, String district, String sex, String age, String arrestDate, String chargeDescription) {
        this.arrest = arrest;
        this.race = race;
        this.district = district;
        this.sex = sex;
        this.age = age;
        this.arrestDate = arrestDate;
        this.chargeDescription = chargeDescription;
    }
    
    public AnonymizedRecord(AnonymizedRecord anonymizedRecord) {
        arrest = anonymizedRecord.getArrest();
        race = anonymizedRecord.getRace();
        district = anonymizedRecord.getDistrict();
        sex = anonymizedRecord.getSex();
        age = anonymizedRecord.getAge();
        arrestDate = anonymizedRecord.getArrestDate();
        chargeDescription = anonymizedRecord.getChargeDescription();
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        AnonymizedRecord that = (AnonymizedRecord) o;
        return Objects.equals(arrest, that.arrest) && Objects.equals(race, that.race) && Objects.equals(district, that.district) && Objects.equals(sex, that.sex) && Objects.equals(age, that.age) && Objects.equals(arrestDate, that.arrestDate) && Objects
                .equals(chargeDescription, that.chargeDescription);
    }

    @Override
    public int hashCode() {
        return Objects.hash(arrest, race, district, sex, age, arrestDate, chargeDescription);
    }

    @Override
    public String toString() {
        return "AnonymizedRecord{" + "arrest='" + arrest + '\'' + ", race='" + race + '\'' + ", district='" + district + '\'' + ", sex='" + sex + '\'' + ", age=" + age + ", arrestDate=" + arrestDate + ", chargeDescription='" + chargeDescription + '\'' + '}';
    }
    
    /*
     * Ψευδο-αναγνωριστικά (Quasi-Identifiers -QI): race, city, sex, age, ArrestDate
     */
    public boolean isEquivalent(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        AnonymizedRecord that = (AnonymizedRecord) o;
        return Objects.equals(age, that.age) &&
                Objects.equals(race, that.race) &&
                Objects.equals(district, that.district) &&
                Objects.equals(sex, that.sex) &&
                Objects.equals(arrestDate, that.arrestDate);
    }
}
