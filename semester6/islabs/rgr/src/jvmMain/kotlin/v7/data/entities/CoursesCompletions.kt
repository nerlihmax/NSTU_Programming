package v7.data.entities

import core.DataRow
import core.TableData
import org.ktorm.entity.Entity
import org.ktorm.schema.Table
import org.ktorm.schema.date
import org.ktorm.schema.int
import java.time.LocalDate

interface CourseCompletion : Entity<CourseCompletion> {
    companion object : Entity.Factory<CourseCompletion>()

    var id: Int
    var employee: Employee
    var course: Course
    var startDate: LocalDate
}

val List<CourseCompletion>.asTableData: TableData
    get() = TableData(
        header = listOf("ID", "Сотрудник", "Курс", "Дата начала"),
        data = map {
            DataRow(
                listOf(
                    it.id.toString(), "${it.employee.name} ${it.employee.surname}", it.course.name,
                    it.startDate.toString()
                )
            )
        }
    )

object CoursesCompletions : Table<CourseCompletion>("courses_completion") {
    var id = int("id").primaryKey().bindTo(CourseCompletion::id)
    var employee = int("employee").references(Employees) { it.employee }
    var course = int("course").references(Courses) { it.course }
    var startDate = date("start_date").bindTo(CourseCompletion::startDate)
}