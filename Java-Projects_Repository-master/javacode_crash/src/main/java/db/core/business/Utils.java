package db.core.business;

import db.core.model.AnonymizedRecord;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

public final class Utils {

    public static List<AnonymizedRecord> copy(List<AnonymizedRecord> records) {
        if (records == null || records.isEmpty()) {
            return Collections.emptyList();
        }

        return records.stream()
                      .map(Utils::copy)
                      .collect(Collectors.toList());
    }

    private static AnonymizedRecord copy(AnonymizedRecord record) {
        var copy = new AnonymizedRecord();
        copy.setArrest(record.getArrest());
        copy.setSex(record.getSex());
        copy.setRace(record.getRace());
        copy.setDistrict(record.getDistrict());
        copy.setAge(record.getAge());
        copy.setArrestDate(record.getArrestDate());
        copy.setChargeDescription(record.getChargeDescription());

        return copy;
    }
}
