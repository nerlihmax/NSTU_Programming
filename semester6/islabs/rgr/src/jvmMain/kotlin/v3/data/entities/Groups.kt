package v3.data.entities

import core.DataRow
import core.TableData
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

val List<Group>.asTableData: TableData
    get() = TableData(
        header = listOf("ID", "Название", "Специальность"),
        data = map { DataRow(listOf(it.id.toString(), it.name, it.specialty)) }
    )

object Groups : Table<Group>("groups") {
    var id = int("id").primaryKey().bindTo(Group::id)
    var name = text("name").bindTo(Group::name)
    var specialty = text("specialty").bindTo(Group::specialty)
}
