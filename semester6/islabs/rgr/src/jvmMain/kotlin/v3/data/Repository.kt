package v3.data

import org.ktorm.database.Database
import org.ktorm.dsl.asc
import org.ktorm.dsl.eq
import org.ktorm.dsl.from
import org.ktorm.dsl.innerJoin
import org.ktorm.dsl.map
import org.ktorm.dsl.orderBy
import org.ktorm.dsl.rightJoin
import org.ktorm.dsl.select
import org.ktorm.dsl.where
import org.ktorm.entity.add
import org.ktorm.entity.find
import org.ktorm.entity.removeIf
import org.ktorm.entity.sequenceOf
import org.ktorm.entity.sortedBy
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
        return database
            .from(TeachersTable)
            .innerJoin(DisciplinesScheduleTable, on = TeachersTable.id eq DisciplinesScheduleTable.teacher)
            .innerJoin(DisciplinesTable, on = DisciplinesScheduleTable.discipline eq DisciplinesTable.id)
            .rightJoin(GroupsTable, on = DisciplinesTable.specialty eq GroupsTable.specialty)
            .select(
                TeachersTable.fullName,
                DisciplinesTable.name,
                GroupsTable.name,
                DisciplinesScheduleTable.hours,
            )
            .orderBy(TeachersTable.id.asc())
            .where(TeachersTable.id eq id)
            .map {
                TeacherDiscipline(
                    name = it[TeachersTable.fullName]!!,
                    discipline = it[DisciplinesTable.name]!!,
                    group = it[GroupsTable.name]!!,
                    hours = it[DisciplinesScheduleTable.hours]!!,
                )
            }
    }

    //указав id дисциплины, получить список преподавателей, ее читающих
    fun getTeachersByDiscipline(id: Int): List<TeacherDTO> {
        return database
            .from(TeachersTable)
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
            .orderBy(TeachersTable.id.asc())
            .map {
                TeacherDTO(
                    id = it[TeachersTable.id]!!,
                    fullName = it[TeachersTable.fullName]!!,
                    post = it[TeachersTable.post]!!,
                    department = database
                        .sequenceOf(DepartmentsTable)
                        .find { dps -> dps.id eq it[DepartmentsTable.id]!! }!!,
                    hireDate = it[TeachersTable.hireDate]!!,
                )
            }
    }

    inner class Teachers {
        fun create(teacher: Teacher): Boolean {
            return database.sequenceOf(TeachersTable).add(teacher) == 1
        }

        fun getAll(): List<TeacherDTO> {
            return database.sequenceOf(TeachersTable).toList().sortedBy { it.id }.map {
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

        fun getByName(name: String): Teacher? {
            return database.sequenceOf(TeachersTable).find { it.fullName eq name }
        }
    }

    inner class Departments {
        fun getAll(): List<Department> {
            return database.sequenceOf(DepartmentsTable).sortedBy { it.id }.toList()
        }

        fun create(department: Department): Boolean {
            return database.sequenceOf(DepartmentsTable).add(department) == 1
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

        fun getByName(name: String): Department? {
            return database.sequenceOf(DepartmentsTable).find { it.name eq name }
        }
    }

    inner class Groups {
        fun getAll(): List<Group> {
            return database.sequenceOf(GroupsTable).sortedBy { it.id }.toList()
        }

        fun create(group: Group): Boolean {
            return database.sequenceOf(GroupsTable).add(group) == 1
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

        fun getByName(name: String): Group? {
            return database.sequenceOf(GroupsTable).find { it.name eq name }
        }
    }

    inner class Disciplines {
        fun getAll(): List<Discipline> {
            return database.sequenceOf(DisciplinesTable).sortedBy { it.id }.toList()
        }

        fun create(discipline: Discipline): Boolean {
            return database.sequenceOf(DisciplinesTable).add(discipline) == 1
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

        fun getByName(name: String): Discipline? {
            return database.sequenceOf(DisciplinesTable).find { it.name eq name }
        }
    }

    inner class DisciplinesSchedule {
        fun getAll(): List<DisciplineSchedule> {
            return database.sequenceOf(DisciplinesScheduleTable).sortedBy { it.id }.toList()
        }

        fun create(disciplineSchedule: DisciplineSchedule): Boolean {
            return database.sequenceOf(DisciplinesScheduleTable).add(disciplineSchedule) == 1
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