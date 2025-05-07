package fastdd.evaluation;

import fastdd.differentialfunction.DifferentialFunction;
import java.util.List;
import java.util.Set;

public class SimpleDifferentialDependency {
    private Set<DifferentialFunction> lhs;
    private DifferentialFunction rhs;

    public SimpleDifferentialDependency(Set<DifferentialFunction> lhs, DifferentialFunction rhs) {
        this.lhs = lhs;
        this.rhs = rhs;
    }

    public Set<DifferentialFunction> getLhs() {
        return lhs;
    }

    public void setLhs(Set<DifferentialFunction> lhs) {
        this.lhs = lhs;
    }

    public DifferentialFunction getRhs() {
        return rhs;
    }

    public void setRhs(DifferentialFunction rhs) {
        this.rhs = rhs;
    }

    @Override
    public String toString() {
        return lhs.toString() + " ⇒ " + rhs.toString();
    }
}
