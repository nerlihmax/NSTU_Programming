package ru.kheynov.hotel.data.entities

import org.ktorm.entity.Entity
import org.ktorm.schema.Table
import org.ktorm.schema.int
import org.ktorm.schema.text

interface Room : Entity<Room> {
    companion object : Entity.Factory<Room>()

    var id: String
    var type: String
    var price: Int
    var hotel: Hotel
}

object Rooms : Table<Room>("rooms") {
    val id = text("id").primaryKey().bindTo { it.id }
    val type = text("type").bindTo { it.type }
    val price = int("price").bindTo { it.price }
    val hotel = int("hotel_id").references(Hotels) { it.hotel }
}