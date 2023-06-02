package v7.data.entities

import core.DataRow
import core.TableData
import org.ktorm.entity.Entity
import org.ktorm.schema.Table
import org.ktorm.schema.date
import org.ktorm.schema.int
import org.ktorm.schema.text
import java.time.LocalDate

interface Employee : Entity<Employee> {
    companion object : Entity.Factory<Employee>()

    var id: Int
    var name: String
    var surname: String
    var department: Department
    var position: Position
    var hireDate: LocalDate
}

val List<Employee>.asTableData: TableData
    get() = TableData(
        header = listOf("ID", "Имя", "Фамилия", "Отдел", "Должность", "Дата приема"),
        data = map {
            DataRow(
                listOf(
                    it.id.toString(), it.name, it.surname, it.department.name, it.position.name, it.hireDate.toString()
                )
            )
        }
    )

object Employees : Table<Employee>("employees") {
    var id = int("id").primaryKey().bindTo(Employee::id)
    var name = text("name").bindTo(Employee::name)
    var surname = text("surname").bindTo(Employee::surname)
    var department = int("department").references(Departments) { it.department }
    var position = int("position").references(Positions) { it.position }
    var hireDate = date("hire_date").bindTo(Employee::hireDate)
}