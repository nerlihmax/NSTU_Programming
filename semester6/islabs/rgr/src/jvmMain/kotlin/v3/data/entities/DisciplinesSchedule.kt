package v3.data.entities

import core.DataRow
import core.TableData
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

val List<DisciplineSchedule>.asTableData: TableData
    get() = TableData(
        header = listOf("ID", "Дисциплина", "Преподаватель", "Часы"),
        data = map { DataRow(listOf(it.id.toString(), it.discipline.name, it.teacher.fullName, it.hours.toString())) }
    )

object DisciplinesSchedule : Table<DisciplineSchedule>("disciplines_schedule") {
    var id = int("id").primaryKey().bindTo(DisciplineSchedule::id)
    var discipline = int("discipline").references(Disciplines) { it.discipline }
    var teacher = int("teacher").references(Teachers) { it.teacher }
    var hours = int("hours").bindTo(DisciplineSchedule::hours)
}
