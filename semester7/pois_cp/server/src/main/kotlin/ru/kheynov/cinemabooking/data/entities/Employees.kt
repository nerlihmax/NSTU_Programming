package ru.kheynov.cinemabooking.data.entities

import org.ktorm.entity.Entity
import org.ktorm.schema.Table
import org.ktorm.schema.text
import ru.kheynov.cinemabooking.data.entities.Cinemas.bindTo
import ru.kheynov.cinemabooking.data.entities.Cinemas.primaryKey

interface Employee : Entity<Employee> {
    companion object : Entity.Factory<Employee>()

    var id: String
    var fullName: String
    var cinema: Cinema
}

object Employees : Table<Employee>("employees") {
    var id = text("id").primaryKey().bindTo(Employee::id)
    var fullName = text("full_name").bindTo(Employee::fullName)
    var cinema = text("cinema").references(Cinemas) { it.cinema }
}