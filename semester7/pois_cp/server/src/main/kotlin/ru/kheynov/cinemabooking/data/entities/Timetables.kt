package ru.kheynov.cinemabooking.data.entities

import org.ktorm.entity.Entity
import org.ktorm.schema.Table
import org.ktorm.schema.int
import org.ktorm.schema.text
import org.ktorm.schema.timestamp
import java.time.Instant

interface Timetable : Entity<Timetable> {
    companion object : Entity.Factory<Timetable>()

    var id: String
    var film: Film
    var cinema: Cinema
    var price: Int
    var time: Instant
}

object Timetables : Table<Timetable>("timetables") {
    var id = text("id").primaryKey().bindTo(Timetable::id)
    var film = text("film").references(Films) { it.film }
    var cinema = text("cinema").references(Cinemas) { it.cinema }
    var price = int("price").bindTo(Timetable::price)
    var time = timestamp("time").bindTo(Timetable::time)
}