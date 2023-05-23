package v3.data.entities

import org.ktorm.entity.Entity
import org.ktorm.schema.Table
import org.ktorm.schema.int
import org.ktorm.schema.text

interface Group : Entity<Group> {
    companion object : Entity.Factory<Group>()

    var id: Int
    var name: String
    var specialty: String
}

object Groups : Table<Group>("groups") {
    var id = int("id").primaryKey().bindTo(Group::id)
    var name = text("name").bindTo(Group::name)
    var specialty = text("specialty").bindTo(Group::specialty)
}
