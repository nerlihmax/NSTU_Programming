package ru.kheynov.cinemabooking.data.entities

import org.ktorm.entity.Entity
import org.ktorm.schema.Table
import org.ktorm.schema.int
import org.ktorm.schema.text

interface Booking : Entity<Booking> {
    companion object : Entity.Factory<Booking>()

    var id: String
    var user: User
    var bookedFilm: Timetable
}

object Bookings : Table<Booking>("bookings") {
    var id = text("id").primaryKey().bindTo(Booking::id)
    var user = int("user_id").references(Users) { it.user }
    var bookedFilm = int("booked_film").references(Timetables) { it.bookedFilm }
}