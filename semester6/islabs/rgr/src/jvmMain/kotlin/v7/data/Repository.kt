package v7.data

import org.ktorm.database.Database
import org.ktorm.dsl.eq
import org.ktorm.dsl.from
import org.ktorm.dsl.innerJoin
import org.ktorm.dsl.map
import org.ktorm.dsl.select
import org.ktorm.entity.add
import org.ktorm.entity.find
import org.ktorm.entity.removeIf
import org.ktorm.entity.sequenceOf
import org.ktorm.entity.toList
import v7.data.entities.Course
import v7.data.entities.CourseCompletion
import v7.data.entities.Department
import v7.data.entities.Employee
import v7.data.entities.Position
import v7.domain.entities.EmployeeCourseInfo
import java.time.LocalDate
import v7.data.entities.Courses as CoursesTable
import v7.data.entities.CoursesCompletions as CoursesCompletionsTable
import v7.data.entities.Departments as DepartmentsTable
import v7.data.entities.Employees as EmployeesTable
import v7.data.entities.Positions as PositionsTable

class Repository(
    private val database: Database,
) {
    private fun getEmployeesCourses(): List<EmployeeCourseInfo> = database
        .from(EmployeesTable)
        .innerJoin(CoursesCompletionsTable, on = EmployeesTable.id eq CoursesCompletionsTable.employee)
        .innerJoin(CoursesTable, on = CoursesCompletionsTable.course eq CoursesTable.id)
        .innerJoin(PositionsTable, on = EmployeesTable.position eq PositionsTable.id)
        .innerJoin(DepartmentsTable, on = EmployeesTable.department eq DepartmentsTable.id)
        .select(
            EmployeesTable.name,
            EmployeesTable.surname,
            PositionsTable.name,
            DepartmentsTable.name,
            CoursesTable.name,
            CoursesTable.description,
            CoursesTable.hours,
            CoursesCompletionsTable.startDate,
        )
        .map {
            EmployeeCourseInfo(
                fullName = "${it[EmployeesTable.name]} ${it[EmployeesTable.surname]}",
                position = it[PositionsTable.name]!!,
                department = it[DepartmentsTable.name]!!,
                course = it[CoursesTable.name]!!,
                courseDescription = it[CoursesTable.description]!!,
                duration = it[CoursesTable.hours]!!,
                startDate = it[CoursesCompletionsTable.startDate]!!,
            )
        }

    fun getEmployeesCurrentCourses(month: Int, year: Int): List<EmployeeCourseInfo> =
        getEmployeesCourses().filter {
            it.startDate.monthValue == month && it.startDate.year == year
        }

    fun getEmployeesPlannedCourses(): List<EmployeeCourseInfo> =
        getEmployeesCourses().filter {
            it.startDate.isBefore(LocalDate.now().plusYears(1))
        }

    fun getPassedEmployeeCoursesByDepartment(id: Int): List<EmployeeCourseInfo> =
        getEmployeesCourses().filter {
            it.department == Departments().getById(id)?.name && it.startDate.isBefore(LocalDate.now())
        }

    inner class Departments {
        fun getAll(): List<Department> {
            return database.sequenceOf(DepartmentsTable).toList().sortedBy { it.id }
        }

        fun create(department: Department): Boolean {
            return database.sequenceOf(DepartmentsTable).add(department) == 1
        }

        fun delete(id: Int): Boolean {
            return database.sequenceOf(DepartmentsTable).removeIf { it.id eq id } == 1
        }

        fun update(department: Department): Boolean {
            return database.sequenceOf(DepartmentsTable).find { it.id eq department.id }?.run {
                name = department.name
                flushChanges()
            } == 1
        }

        fun getByName(name: String): Department? {
            return database.sequenceOf(DepartmentsTable).find { it.name eq name }
        }

        fun getById(id: Int): Department? {
            return database.sequenceOf(DepartmentsTable).find { it.id eq id }
        }
    }

    inner class Positions {
        fun getAll(): List<Position> {
            return database.sequenceOf(PositionsTable).toList().sortedBy { it.id }
        }

        fun create(position: Position): Boolean {
            return database.sequenceOf(PositionsTable).add(position) == 1
        }

        fun delete(id: Int): Boolean {
            return database.sequenceOf(PositionsTable).removeIf { it.id eq id } == 1
        }

        fun update(position: Position): Boolean {
            return database.sequenceOf(PositionsTable).find { it.id eq position.id }?.run {
                name = position.name
                flushChanges()
            } == 1
        }

        fun getByName(name: String): Position? {
            return database.sequenceOf(PositionsTable).find { it.name eq name }
        }
    }

    inner class Courses {
        fun getAll(): List<Course> {
            return database.sequenceOf(CoursesTable).toList().sortedBy { it.id }
        }

        fun create(course: Course): Boolean {
            return database.sequenceOf(CoursesTable).add(course) == 1
        }

        fun delete(id: Int): Boolean {
            return database.sequenceOf(CoursesTable).removeIf { it.id eq id } == 1
        }

        fun update(course: Course): Boolean {
            return database.sequenceOf(CoursesTable).find { it.id eq course.id }?.run {
                name = course.name
                department = course.department
                hours = course.hours
                description = course.description
                flushChanges()
            } == 1
        }

        fun getByName(name: String): Course? {
            return database.sequenceOf(CoursesTable).find { it.name eq name }
        }
    }

    inner class Employees {
        fun getAll(): List<Employee> {
            return database.sequenceOf(EmployeesTable).toList().sortedBy { it.id }
        }

        fun create(employee: Employee): Boolean {
            return database.sequenceOf(EmployeesTable).add(employee) == 1
        }

        fun delete(id: Int): Boolean {
            return database.sequenceOf(EmployeesTable).removeIf { it.id eq id } == 1
        }

        fun update(employee: Employee): Boolean {
            return database.sequenceOf(EmployeesTable).find { it.id eq employee.id }?.run {
                name = employee.name
                surname = employee.surname
                position = employee.position
                department = employee.department
                hireDate = employee.hireDate
                flushChanges()
            } == 1
        }

        fun getByName(name: String): Employee? {
            return database.sequenceOf(EmployeesTable).find { it.name eq name }
        }
    }

    inner class CoursesCompletions {
        fun getAll(): List<CourseCompletion> {
            return database.sequenceOf(CoursesCompletionsTable).toList().sortedBy { it.id }
        }

        fun create(courseCompletion: CourseCompletion): Boolean {
            return database.sequenceOf(CoursesCompletionsTable).add(courseCompletion) == 1
        }

        fun delete(id: Int): Boolean {
            return database.sequenceOf(CoursesCompletionsTable).removeIf { it.id eq id } == 1
        }

        fun update(courseCompletion: CourseCompletion): Boolean {
            return database.sequenceOf(CoursesCompletionsTable).find { it.id eq courseCompletion.id }?.run {
                employee = courseCompletion.employee
                course = courseCompletion.course
                startDate = courseCompletion.startDate
                flushChanges()
            } == 1
        }
    }
}