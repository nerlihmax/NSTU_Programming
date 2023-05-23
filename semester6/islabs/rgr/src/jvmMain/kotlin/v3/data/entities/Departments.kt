package v3.data.entities

import org.ktorm.entity.Entity
import org.ktorm.schema.Table
import org.ktorm.schema.int
import org.ktorm.schema.text

interface Department : Entity<Department> {
    companion object : Entity.Factory<Department>()

    var id: Int
    var name: String
}

object Departments : Table<Department>("departments") {
    var id = int("id").primaryKey().bindTo(Department::id)
    var name = text("name").bindTo(Department::name)
}
