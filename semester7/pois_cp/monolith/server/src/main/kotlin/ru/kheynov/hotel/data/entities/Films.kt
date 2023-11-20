package ru.kheynov.cinemabooking.data.entities

import org.ktorm.entity.Entity
import org.ktorm.schema.Table
import org.ktorm.schema.int
import org.ktorm.schema.text

interface Film : Entity<Film> {
    companion object : Entity.Factory<Film>()

    var id: Int
    var name: String
    var duration: Int
    var image: String?
}

object Films : Table<Film>("films") {
    var id = int("id").primaryKey().bindTo(Film::id)
    var name = text("name").bindTo(Film::name)
    var duration = int("duration").bindTo(Film::duration)
    var image = text("image").bindTo(Film::image)
}
