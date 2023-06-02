package v3.data.entities

import core.DataRow
import core.TableData
import org.ktorm.entity.Entity
import org.ktorm.schema.Table
import org.ktorm.schema.date
import org.ktorm.schema.int
import org.ktorm.schema.text
import java.time.LocalDate

interface Teacher : Entity<Teacher> {
    companion object : Entity.Factory<Teacher>()

    var id: Int
    var fullName: String
    var department: Department
    var post: String
    var hireDate: LocalDate
}

object Teachers : Table<Teacher>("teachers") {
    var id = int("id").primaryKey().bindTo(Teacher::id)
    var fullName = text("full_name").bindTo(Teacher::fullName)
    var department = int("department").references(Departments) { it.department }
    var post = text("post").bindTo(Teacher::post)
    var hireDate = date("hire_date").bindTo(Teacher::hireDate)
}
