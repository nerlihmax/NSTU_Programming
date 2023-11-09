package ru.kheynov.cinemabooking.data.entities

import org.ktorm.entity.Entity
import org.ktorm.schema.Table
import org.ktorm.schema.int
import org.ktorm.schema.text

interface Cinema : Entity<Cinema> {
    companion object : Entity.Factory<Cinema>()

    var id: Int
    var name: String
    var city: String
    var address: String?
}

object Cinemas : Table<Cinema>("cinemas") {
    var id = int("id").primaryKey().bindTo(Cinema::id)
    var name = text("name").bindTo(Cinema::name)
    var city = text("city").bindTo(Cinema::city)
    var address = text("address").bindTo(Cinema::address)
}