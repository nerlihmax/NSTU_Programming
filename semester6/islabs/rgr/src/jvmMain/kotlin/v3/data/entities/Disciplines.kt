package v3.data.entities

import org.ktorm.entity.Entity
import org.ktorm.schema.Table
import org.ktorm.schema.int
import org.ktorm.schema.text

interface Discipline : Entity<Discipline> {
    companion object : Entity.Factory<Discipline>()

    var id: Int
    var name: String
    var semester: Int
    var specialty: String
}

object Disciplines : Table<Discipline>("disciplines") {
    var id = int("id").primaryKey().bindTo(Discipline::id)
    var name = text("name").bindTo(Discipline::name)
    var semester = int("semester").bindTo(Discipline::semester)
    var specialty = text("specialty").bindTo(Discipline::specialty)
}
