package v3.data

import org.ktorm.database.Database
import org.ktorm.dsl.eq
import org.ktorm.dsl.from
import org.ktorm.dsl.innerJoin
import org.ktorm.dsl.map
import org.ktorm.dsl.rightJoin
import org.ktorm.dsl.select
import org.ktorm.dsl.where
import org.ktorm.entity.add
import org.ktorm.entity.find
import org.ktorm.entity.removeIf
import org.ktorm.entity.sequenceOf
import org.ktorm.entity.toList
import v3.data.entities.Department
import v3.data.entities.Discipline
import v3.data.entities.DisciplineSchedule
import v3.data.entities.Group
import v3.data.entities.Teacher
import v3.domain.entities.TeacherDiscipline
import v3.data.entities.Departments as DepartmentsTable
import v3.data.entities.Disciplines as DisciplinesTable
import v3.data.entities.DisciplinesSchedule as DisciplinesScheduleTable
import v3.data.entities.Groups as GroupsTable
import v3.data.entities.Teachers as TeachersTable
import v3.domain.entities.Department as DepartmentDTO
import v3.domain.entities.Teacher as TeacherDTO

class Repository(
    private val database: Database,
) {
    //выдать справки по преподавателю - какие дисциплины он читает, для каких групп и сколько часов
    fun getTeacherInfo(id: Int): List<TeacherDiscipline> {
        return database.from(TeachersTable)
            .innerJoin(DisciplinesScheduleTable, on = TeachersTable.id eq DisciplinesScheduleTable.teacher)
            .innerJoin(DisciplinesTable, on = DisciplinesScheduleTable.discipline eq DisciplinesTable.id)
            .rightJoin(GroupsTable, on = DisciplinesTable.specialty eq GroupsTable.specialty)
            .select(
                TeachersTable.fullName,
                DisciplinesTable.name,
                GroupsTable.name,
                DisciplinesScheduleTable.hours,
            )
            .where(TeachersTable.id eq id)
            .map {
                TeacherDiscipline(
                    name = it[TeachersTable.fullName]!!,
                    discipline = v3.domain.entities.Discipline(
                        id = it[DisciplinesTable.id]!!,
                        name = it[DisciplinesTable.name]!!,
                        semester = it[DisciplinesTable.semester]!!,
                        specialty = it[DisciplinesTable.specialty]!!,
                    ),
                    group = v3.domain.entities.Group(
                        id = it[GroupsTable.id]!!,
                        name = it[GroupsTable.name]!!,
                        specialty = it[GroupsTable.specialty]!!,
                    ),
                    hours = it[DisciplinesScheduleTable.hours]!!,
                )
            }
    }

    //указав id дисциплины, получить список преподавателей, ее читающих
    fun getTeachersByDiscipline(id: Int): List<TeacherDTO> {
        return database.from(TeachersTable)
            .innerJoin(DisciplinesScheduleTable, on = TeachersTable.id eq DisciplinesScheduleTable.teacher)
            .innerJoin(DisciplinesTable, on = DisciplinesScheduleTable.discipline eq DisciplinesTable.id)
            .innerJoin(DepartmentsTable, on = TeachersTable.department eq DepartmentsTable.id)
            .select(
                TeachersTable.id,
                TeachersTable.fullName,
                DepartmentsTable.id,
                TeachersTable.post,
                TeachersTable.hireDate,
            )
            .where(DisciplinesTable.id eq id)
            .map {
                TeacherDTO(
                    id = it[TeachersTable.id]!!,
                    fullName = it[TeachersTable.fullName]!!,
                    post = it[TeachersTable.post]!!,
                    department = database.sequenceOf(DepartmentsTable)
                        .find { dps -> dps.id eq it[DepartmentsTable.id]!! }!!,
                    hireDate = it[TeachersTable.hireDate]!!,
                )
            }
    }

    inner class Teachers {
        fun create(teacher: TeacherDTO): Boolean {
            val entity = Teacher {
                fullName = teacher.fullName
                department = teacher.department
                post = teacher.post
                hireDate = teacher.hireDate
            }
            return database.sequenceOf(TeachersTable).add(entity) == 1
        }

        fun getAll(): List<TeacherDTO> {
            return database.sequenceOf(TeachersTable).toList().map {
                TeacherDTO(
                    id = it.id,
                    fullName = it.fullName,
                    department = it.department,
                    post = it.post,
                    hireDate = it.hireDate,
                )
            }
        }

        fun delete(id: Int): Boolean {
            return database.sequenceOf(TeachersTable).removeIf { it.id eq id } == 1
        }

        fun update(teacher: TeacherDTO): Boolean {
            return database.sequenceOf(TeachersTable).find { it.id eq teacher.id }?.run {
                fullName = teacher.fullName
                department = teacher.department
                post = teacher.post
                hireDate = teacher.hireDate
                flushChanges()
            } == 1
        }

        fun getById(id: Int): TeacherDTO? {
            return database.sequenceOf(TeachersTable).find { it.id eq id }?.run {
                TeacherDTO(
                    id = id,
                    fullName = fullName,
                    department = department,
                    post = post,
                    hireDate = hireDate,
                )
            }
        }
    }

    inner class Departments {
        fun getAll(): List<Department> {
            return database.sequenceOf(DepartmentsTable).toList()
        }

        fun create(department: DepartmentDTO): Boolean {
            val entity = Department {
                name = department.name
            }
            return database.sequenceOf(DepartmentsTable).add(entity) == 1
        }

        fun delete(id: Int): Boolean {
            return database.sequenceOf(DepartmentsTable).removeIf { it.id eq id } == 1
        }

        fun update(department: DepartmentDTO): Boolean {
            return database.sequenceOf(DepartmentsTable).find { it.id eq department.id }?.run {
                name = department.name
                flushChanges()
            } == 1
        }

        fun getById(id: Int): DepartmentDTO? {
            return database.sequenceOf(DepartmentsTable).find { it.id eq id }?.run {
                DepartmentDTO(
                    id = id,
                    name = name,
                )
            }
        }
    }

    inner class Groups {
        fun getAll(): List<Group> {
            return database.sequenceOf(GroupsTable).toList()
        }

        fun create(group: Group): Boolean {
            val entity = Group {
                name = group.name
                specialty = group.specialty
            }
            return database.sequenceOf(GroupsTable).add(entity) == 1
        }

        fun delete(id: Int): Boolean {
            return database.sequenceOf(GroupsTable).removeIf { it.id eq id } == 1
        }

        fun update(group: Group): Boolean {
            return database.sequenceOf(GroupsTable).find { it.id eq group.id }?.run {
                name = group.name
                specialty = group.specialty
                flushChanges()
            } == 1
        }
    }

    inner class Disciplines {
        fun getAll(): List<Discipline> {
            return database.sequenceOf(DisciplinesTable).toList()
        }

        fun create(discipline: Discipline): Boolean {
            val entity = Discipline {
                name = discipline.name
                specialty = discipline.specialty
                semester = discipline.semester
            }
            return database.sequenceOf(DisciplinesTable).add(entity) == 1
        }

        fun delete(id: Int): Boolean {
            return database.sequenceOf(DisciplinesTable).removeIf { it.id eq id } == 1
        }

        fun update(discipline: Discipline): Boolean {
            return database.sequenceOf(DisciplinesTable).find { it.id eq discipline.id }?.run {
                name = discipline.name
                specialty = discipline.specialty
                semester = discipline.semester
                flushChanges()
            } == 1
        }
    }

    inner class DisciplinesSchedule {
        fun getAll(): List<DisciplineSchedule> {
            return database.sequenceOf(DisciplinesScheduleTable).toList()
        }

        fun create(disciplineSchedule: DisciplineSchedule): Boolean {
            val entity = DisciplineSchedule {
                discipline = disciplineSchedule.discipline
                teacher = disciplineSchedule.teacher
                hours = disciplineSchedule.hours
            }
            return database.sequenceOf(DisciplinesScheduleTable).add(entity) == 1
        }

        fun delete(id: Int): Boolean {
            return database.sequenceOf(DisciplinesScheduleTable).removeIf { it.id eq id } == 1
        }

        fun update(disciplineSchedule: DisciplineSchedule): Boolean {
            return database.sequenceOf(DisciplinesScheduleTable).find { it.id eq disciplineSchedule.id }?.run {
                discipline = disciplineSchedule.discipline
                teacher = disciplineSchedule.teacher
                hours = disciplineSchedule.hours
                flushChanges()
            } == 1
        }
    }
}