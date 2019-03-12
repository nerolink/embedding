package com.yjy.constants;

/**
 * Created by DJH on 2018/9/10
 */
public final class Collection {

    /*
    采集数据集合
     */
    public static final String ANSWER = "Answer";
    public static final String ATTENDANCE = "Attendance";
    public static final String COURSEWARE = "Courseware";
    public static final String DEVICE_INFO = "DeviceInfo";
    public static final String FILES_DISTRIBUTED = "FilesDistributed";
    public static final String GROUP_EXIST = "GroupExist";
    public static final String LESSON = "Lesson";
    public static final String LOCATION_INFO = "LocationInfo";
    public static final String QUESTION = "Question";
    public static final String SCORE = "Score";
    public static final String SEQUENCE = "Sequence";
    public static final String SHARE = "Share";
    public static final String VOTE = "Vote";
    public static final String WHITEBOARD = "Whiteboard";
    public static final String WORKS = "Works";
    /*
    for class集合
     */
    public static final String XML_ACTIVITY = "Xml_Activity";
    public static final String XML_ANSWER = "Xml_Answer";
    public static final String XML_LESSON = "Xml_Lesson";
    public static final String XML_MODULE = "Xml_Module";
    public static final String XML_SCHOOL = "Xml_School";
    public static final String XML_SEQUENCE = "Xml_Sequence";

    private Collection() {
        throw new RuntimeException();
    }

    public static class Xml_Lesson {
        public static final String FIELD_CLASSID = "object.definition.description.classid";
        public static final String FIELD_SUBJECT = "object.definition.description.subject";
        public static final String FIELD_CLASS = "object.definition.description.class";
        public static final String FIELD_COURSE_ID = "object.definition.description.courseId";
        public static final String FIELD_END_TIME = "object.definition.description.endTime";
        public static final String FIELD_SECTION = "object.definition.description.section";
        public static final String FIELD_START_TIME = "object.definition.description.startTime";
        public static final String FIELD_UNIT = "object.definition.description.unit";
        public static final String FIELD_TEXTBOOK_TITLE = "object.definition.description.textbookTitle";
        public static final String FIELD_LESSON_NAME = "object.definition.description.lessonName";
        public static final String FIELD_SCHOOL = "object.definition.description.school";
        public static final String FIELD_GRADE = "object.definition.description.grade";
        public static final String FIELD_DISCIPLINE = "object.definition.description.discipline";
        public static final String FIELD_ACTOR = "actor.id";
    }
}
