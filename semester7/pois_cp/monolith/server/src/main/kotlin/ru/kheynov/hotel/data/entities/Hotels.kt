package ru.kheynov.hotel.data.entities

import org.ktorm.entity.Entity
import org.ktorm.schema.Table
import org.ktorm.schema.int
import org.ktorm.schema.text

interface Hotel : Entity<Hotel> {
    companion object : Entity.Factory<Hotel>()

    var id: Int
    var name: String
    var city: String
    var address: String
    var rating: Int
}

object Hotels : Table<Hotel>("hotels") {
    var id = int("id").primaryKey().bindTo(Hotel::id)
    var name = text("name").bindTo(Hotel::name)
    var city = text("city").bindTo(Hotel::city)
    var address = text("address").bindTo(Hotel::address)
    var rating = int("rating").bindTo(Hotel::rating)
}