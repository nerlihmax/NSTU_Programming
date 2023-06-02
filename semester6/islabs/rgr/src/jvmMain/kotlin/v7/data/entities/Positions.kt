package v7.data.entities

import core.DataRow
import core.TableData
import org.ktorm.entity.Entity
import org.ktorm.schema.Table
import org.ktorm.schema.int
import org.ktorm.schema.text

interface Position : Entity<Position> {
    companion object : Entity.Factory<Position>()

    var id: Int
    var name: String
}

val List<Position>.asTableData: TableData
    get() = TableData(
        header = listOf("ID", "Название"),
        data = map { DataRow(listOf(it.id.toString(), it.name)) }
    )

object Positions : Table<Position>("positions") {
    var id = int("id").primaryKey().bindTo(Position::id)
    var name = text("name").bindTo(Position::name)
}
