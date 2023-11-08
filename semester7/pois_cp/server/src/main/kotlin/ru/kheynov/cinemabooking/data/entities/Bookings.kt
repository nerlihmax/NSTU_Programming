package ru.kheynov.cinemabooking.data.entities

import org.ktorm.entity.Entity
import org.ktorm.schema.Table
import org.ktorm.schema.text

interface Booking : Entity<Booking> {
    companion object : Entity.Factory<Booking>()

    var id: String
    var employee: Employee
    var bookedFilm: Timetable
}

object Bookings : Table<Booking>("bookings") {
    var id = text("id").primaryKey().bindTo(Booking::id)
    var employee = text("employee").references(Employees) { it.employee }
    var bookedFilm = text("booked_film").references(Timetables) { it.bookedFilm }
}