package v7.data.entities

import core.DataRow
import core.TableData
import org.ktorm.entity.Entity
import org.ktorm.schema.Table
import org.ktorm.schema.int
import org.ktorm.schema.text

interface Course : Entity<Course> {
    companion object : Entity.Factory<Course>()

    var id: Int
    var name: String
    var department: Department
    var hours: Int
    var description: String
}

val List<Course>.asTableData: TableData
    get() = TableData(
        header = listOf("ID", "Название", "Отдел", "Часы", "Описание"),
        data = map {
            DataRow(
                listOf(it.id.toString(), it.name, it.department.name, it.hours.toString(), it.description)
            )
        }
    )

object Courses : Table<Course>("courses") {
    var id = int("id").primaryKey().bindTo(Course::id)
    var name = text("name").bindTo(Course::name)
    var department = int("department").references(Departments) { it.department }
    var hours = int("hours").bindTo(Course::hours)
    var description = text("description").bindTo(Course::description)
}
