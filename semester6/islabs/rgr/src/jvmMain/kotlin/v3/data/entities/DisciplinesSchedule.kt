package v3.data.entities

import org.ktorm.entity.Entity
import org.ktorm.schema.Table
import org.ktorm.schema.int

interface DisciplineSchedule : Entity<DisciplineSchedule> {
    companion object : Entity.Factory<DisciplineSchedule>()

    var id: Int
    var discipline: Discipline
    var teacher: Teacher
    var hours: Int
}

object DisciplinesSchedule : Table<DisciplineSchedule>("disciplines_schedule") {
    var id = int("id").primaryKey().bindTo(DisciplineSchedule::id)
    var discipline = int("discipline_id").references(Disciplines) { it.discipline }
    var teacher = int("teacher_id").references(Teachers) { it.teacher }
    var hours = int("hours").bindTo(DisciplineSchedule::hours)
}
