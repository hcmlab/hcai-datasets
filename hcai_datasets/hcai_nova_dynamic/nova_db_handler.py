import os
import errno
import copy
import warnings
import configparser

from typing import Union
from datetime import datetime
from pymongo import MongoClient
from pymongo.results import InsertOneResult, UpdateResult


class NovaDBHandler:
    ANNOTATOR_COLLECTION = "Annotators"
    SCHEME_COLLECTION = "Schemes"
    STREAM_COLLECTION = "Streams"
    ROLE_COLLECTION = "Roles"
    ANNOTATION_COLLECTION = "Annotations"
    SESSION_COLLECTION = "Sessions"
    ANNOTATION_DATA_COLLECTION = "AnnotationData"

    def __init__(self, db_config_path=None, db_config_dict=None):

        # Connecting to the database
        if db_config_path:
            if os.path.isfile(db_config_path):
                cfg = self.read_config(db_config_path)
                self.ip = str(cfg["DB"]["ip"])
                self.port = int(cfg["DB"]["port"])
                self.user = str(cfg["DB"]["user"])
                self.password = str(cfg["DB"]["password"])
            else:
                raise FileNotFoundError(
                    "No database config file found at {}".format(db_config_path)
                )

        # If a db config_dict is specified overwrite the config from path
        if db_config_dict:
            if db_config_path:
                print(
                    "WARNING! database config are specifed as file AND and as dictionary. Using the dictionary."
                )
            self.ip = db_config_dict["ip"]
            self.port = db_config_dict["port"]
            self.user = db_config_dict["user"]
            self.password = db_config_dict["password"]

        if not (self.ip or self.port or self.user or self.password):
            print(
                "WARNING! No valid nova database config found for path {} and dict {} \n Found config parameters are ip:{}, port{}, user: {}. Also check your password.".format(
                    db_config_path, db_config_dict, self.ip, self.port, self.user
                )
            )

        self.client = MongoClient(
            host=self.ip, port=self.port, username=self.user, password=self.password
        )

    def print_config(self, cfg, cfg_path):
        print("Loaded config from {}:".format(cfg_path))
        print("---------------------")
        for sec_name, sec_dict in cfg._sections.items():
            print(sec_name)
            for k, v in sec_dict.items():
                if k == "password":
                    continue
                else:
                    print("\t{} : {}".format(k, v))
        print("---------------------")

    def read_config(self, cfg_path):
        config = configparser.RawConfigParser()
        config.read(cfg_path)
        self.print_config(config, cfg_path)
        return config

    # Reading from database
    def get_docs_by_prop(
            self, vals: Union[list, str], property: str, database: str, collection: str
    ) -> list:
        """
        Fetching a document from the mongo db collection in the respective database where the passed values are matching a specific property in the collection.

        Args:
          vals: The specific value of a property in the document
          property: The property to look for the passed values
          database: The name of the database to search
          collection: The name of the collection within the database to search

        Returns:
          list: List of documents that match the specified criteria

        """
        filter = []

        if not type(vals) == type(list()):
            vals = [vals]

        for n in vals:
            filter.append({property: n})

        filter = {"$or": filter}
        ret = list(self.client[database][collection].find(filter))
        return ret

    def get_schemes(self, dataset, schemes):
        """
        Fetches the scheme object that matches the specified criteria from the nova database and returns them as a python readable dictionary.
        Args:
          ip:
          port:
          user:
          password:
          dataset:
          scheme:

        Returns:

        """

        if not schemes:
            print("WARNING: No Schemes have been requested. Returning empty list.")
            return []

        # if not dataset in self.datasets:
        # raise ValueError('{} not found in datasets'.format(dataset))

        mongo_schemes = []
        for scheme in schemes:
            mongo_scheme = self.get_docs_by_prop(
                scheme, "name", dataset, self.SCHEME_COLLECTION
            )
            if not mongo_scheme:
                print(f"WARNING: No scheme {scheme} found in database for dataset {dataset}")
            else:
                mongo_schemes.append(mongo_scheme[0])

        if not mongo_schemes:
            raise ValueError(
                "No entries for schemes {} found in database".format(schemes)
            )

        return mongo_schemes

    def get_session_info(self, dataset, session):
        """
        Fetches the session object that matches the specified criteria from the nova database and returns them as a python readable dictionary.

        Args:
          dataset:
          session:

        Returns:

        """
        mongo_session = self.get_docs_by_prop(
            session, "name", dataset, self.SESSION_COLLECTION
        )
        return mongo_session

    def get_data_streams(self, dataset, data_streams):
        """
        Fetches the stream objects that matches the specified criteria from the nova database and returns them as a python readable dictionary.
        Args:
          dataset:
          session:
          role_list:
          data_stream_list:
        """
        # if not dataset in self.datasets:
        #  raise ValueError('{} not found in datasets'.format(dataset))

        if not data_streams:
            print("WARNING: No Datastreams have been requested. Returning empty list.")
            return []

        mongo_streams = []
        for stream in data_streams:
            mongo_stream = self.get_docs_by_prop(
                stream, "name", dataset, self.STREAM_COLLECTION
            )
            if not mongo_stream:
                print("WARNING: No stream {} found in database".format(stream))
            else:
                mongo_streams.append(mongo_stream[0])

        if not mongo_streams:
            raise ValueError("no entries for datastream {} found".format(data_streams))

        return mongo_streams

    def get_annotation_docs(
            self,
            mongo_schemes,
            mongo_sessions,
            mongo_annotators,
            mongo_roles,
            database,
            collection,
    ):
        """
        Fetches all annotationobjects that match the specified criteria from the nova database
        Args:
          mongo_schemes:
          mongo_sessions:
          mongo_annotators:
          mongo_roles:
          database:
          collection:
          client:

        Returns:

        """
        scheme_filter = []
        role_filter = []
        annotator_filter = []
        session_filter = []

        for ms in mongo_schemes:
            scheme_filter.append({"scheme_id": ms["_id"]})

        for mse in mongo_sessions:
            session_filter.append({"session_id": mse["_id"]})

        for ma in mongo_annotators:
            annotator_filter.append({"annotator_id": ma["_id"]})

        for mr in mongo_roles:
            role_filter.append({"role_id": mr["_id"]})

        filter = {
            "$and": [
                {"$or": scheme_filter},
                {"$or": session_filter},
                {"$or": role_filter},
                {"$or": annotator_filter},
            ]
        }

        ret = list(self.client[database][collection].find(filter))
        return ret

    def get_annos(
            self,
            dataset: str,
            scheme: str,
            session: str,
            annotator: str,
            roles: Union[list, str],
    ) -> list:
        """
        Fetches all annotations that matches the specified criteria from the nova database and returns them as a list of python readable dictionaries.
        Args:
          dataset:
          scheme:
          session:
          annotator:
          roles:

        Returns:

        """
        mongo_schemes = self.get_docs_by_prop(
            scheme, "name", dataset, self.SCHEME_COLLECTION
        )
        if not mongo_schemes:
            warnings.warn(f"Unknown scheme {scheme} found")
            return []
        mongo_annotators = self.get_docs_by_prop(
            annotator, "name", dataset, self.ANNOTATOR_COLLECTION
        )
        if not mongo_annotators:
            warnings.warn(f"Unknown annotator {annotator} found")
            return []
        mongo_roles = self.get_docs_by_prop(roles, "name", dataset, self.ROLE_COLLECTION)
        if not mongo_roles:
            warnings.warn(f"Unknown role {roles} found")
            return []
        mongo_sessions = self.get_docs_by_prop(
            session, "name", dataset, self.SESSION_COLLECTION
        )
        if not mongo_sessions:
            warnings.warn(f"Unknown for session {session} found")
            return []

        mongo_annos = self.get_annotation_docs(
            mongo_schemes,
            mongo_sessions,
            mongo_annotators,
            mongo_roles,
            dataset,
            self.ANNOTATION_COLLECTION,
        )

        # getting the annotation data and the session name
        if not mongo_annos:
            raise FileNotFoundError(errno.ENOENT, 'No such annotation', f"annotator: {annotator} - scheme: {scheme} - session: {session} - role: {roles}")

        else:
            # TODO: adapt for multiple roles, annotators etc.
            label = self.get_data_docs_by_prop(
                mongo_annos[0]["data_id"], "_id", dataset
            )
            label = label["labels"]

        return label

    def insert_doc_by_prop(
            self, doc: dict, database: str, collection: str
    ) -> InsertOneResult:
        """
        Uploading a document to the database using the specificed parameters
        Args:
          docs: List of dictionaries with objects to upload to the database
          database: The name of the database to search
          collection: The name of the collection within the database to search

        Returns:
          str: ObjectID of the inserted objects or an empty list in case of failure
        """
        ret = self.client[database][collection].insert_one(doc)
        return ret

    def update_doc_by_prop(
            self, doc: dict, database: str, collection: str
    ) -> UpdateResult:
        """
        Uploading a document to the database using the specificed parameters
        Args:
          docs: List of dictionaries with objects to upload to the database
          database: The name of the database to search
          collection: The name of the collection within the database to search

        Returns:
          str: ObjectID of the inserted objects or an empty list in case of failure
        """
        ret = self.client[database][collection].update_one(
            {"_id": doc["_id"]}, {"$set": doc}
        )
        return ret

    def update_doc_by_id(
            self, _id: str, doc: dict, database: str, collection: str
    ) -> UpdateResult:
        """
        Uploading a document to the database using the specificed parameters
        Args:
          _id: ID of doc, which has to be updated
          docs: List of dictionaries with objects to upload to the database
          database: The name of the database to search
          collection: The name of the collection within the database to search

        Returns:
          str: ObjectID of the inserted objects or an empty list in case of failure
        """
        ret = self.client[database][collection].update_one(
            {"_id": _id}, {"$set": doc}
        )
        return ret

    # TODO: Remove Restclass Labels in discrete Cases
    # TODO: Consider "forced overwrite"
    # TODO: Add Backup case
    # TODO: Call preprocess of annotation
    def set_annos(
            self,
            database: str,
            scheme: str,
            session: str,
            annotator: str,
            role: str,
            annos: list,
            is_finished: bool = False,
            is_locked: bool = False,
    ) -> str:
        """
        Uploading annotations to the database
        Args:
          database:
          scheme:
          session:
          annotator:
          role:
          annos:

        Returns: Object ID of the inserted annotations. Empty string in case of failure
        """
        mongo_scheme = self.get_mongo_scheme(scheme, database)
        mongo_annotator = self.get_mongo_annotator(annotator, database)
        mongo_role = self.get_mongo_role(role, database)
        mongo_session = self.get_mongo_session(session, database)

        # Check if annotations already exist
        mongo_annos = self.get_annotation_docs(
            mongo_scheme,
            mongo_session,
            mongo_annotator,
            mongo_role,
            database,
            self.ANNOTATION_COLLECTION,
        )

        # Check for existing annotations
        mongo_anno_id = None
        mongo_data_id = None
        if mongo_annos:
            if mongo_annos[0]["isLocked"]:
                warnings.warn(f"Can't overwrite locked annotation {str(mongo_annos[0]['_id'])}")
                return ""
            else:
                warnings.warn(f"Overwriting existing annotation {str(mongo_annos[0]['_id'])}")
                mongo_anno_id = mongo_annos[0]["_id"]
                mongo_data_id = mongo_annos[0]["data_id"]

        # Upload label data
        mongo_label_doc = {"labels": annos}
        if mongo_data_id:
            mongo_label_doc["_id"] = mongo_data_id
            success = self.update_doc_by_prop(
                doc=mongo_label_doc,
                database=database,
                collection=self.ANNOTATION_DATA_COLLECTION,
            )
            if not success.acknowledged:
                warnings.warn(f"Unknown error update database entries for Annotation data {mongo_data_id}")
                return ""
            else:
                data_id = mongo_data_id
        else:
            success = self.insert_doc_by_prop(
                doc=mongo_label_doc,
                database=database,
                collection=self.ANNOTATION_DATA_COLLECTION,
            )
            if not success.acknowledged:
                warnings.warn(f"Unexpected error uploading annotation data for {database} - {session} - {scheme} - "
                              f"{annotator}. Upload failed.")
                return ""
            else:
                data_id = success.inserted_id

        # Upload annotation information
        mongo_anno_doc = {
            "data_id": data_id,
            "annotator_id": mongo_annotator[0]["_id"],
            "role_id": mongo_role[0]["_id"],
            "scheme_id": mongo_scheme[0]["_id"],
            "session_id": mongo_session[0]["_id"],
            "isFinished": is_finished,
            "isLocked": is_locked,
            "date": datetime.today().replace(microsecond=0),
        }

        if mongo_anno_id:
            mongo_anno_doc["_id"] = mongo_anno_id
            success = self.update_doc_by_prop(
                doc=mongo_anno_doc, database=database, collection=self.ANNOTATION_COLLECTION
            )
            if not success.acknowledged:
                warnings.warn(
                    f"Unexpected error uploading annotations for {database} - {session} - {scheme} - {annotator}. Upload failed."
                )
                return ""
            else:
                anno_id = mongo_anno_id
        else:
            success = self.insert_doc_by_prop(
                doc=mongo_anno_doc, database=database, collection=self.ANNOTATION_COLLECTION
            )
            if not success.acknowledged:
                warnings.warn(
                    f"Unexpected error uploading annotations for {database} - {session} - {scheme} - {annotator}. Upload failed."
                )
                return ""
            else:
                anno_id = success.inserted_id
        return anno_id

    def get_mongo_scheme(self, scheme, database):
        mongo_scheme = self.get_docs_by_prop(scheme, "name", database, self.SCHEME_COLLECTION)
        if not mongo_scheme:
            warnings.warn(f"Unknown scheme {scheme} found")
            return ""

        return mongo_scheme

    def get_mongo_annotator(self, annotator, database):
        mongo_annotator = self.get_docs_by_prop(annotator, "name", database, self.ANNOTATOR_COLLECTION)
        if not mongo_annotator:
            warnings.warn(f"Unknown annotator {annotator} found")
            return ""

        return mongo_annotator

    def get_mongo_role(self, role, database):
        mongo_role = self.get_docs_by_prop(role, "name", database, self.ROLE_COLLECTION)
        if not mongo_role:
            warnings.warn(f"Unknown role {role} found")
            return ""

        return mongo_role

    def get_mongo_session(self, session, database):
        mongo_session = self.get_docs_by_prop(session, "name", database, self.SESSION_COLLECTION)
        if not mongo_session:
            warnings.warn(f"Unknown for session {session} found")
            return ""

        return mongo_session

    def delete_doc_with_tail(self, doc_id_to_remove, database):
        while doc_id_to_remove is not None:
            remove_id = copy.deepcopy(doc_id_to_remove)
            result = self.get_fields_by_properties(doc_id_to_remove, "_id", "nextEntry", database,
                                                   self.ANNOTATION_DATA_COLLECTION)
            if result is not None and 'nextEntry' in result:
                doc_id_to_remove = result['nextEntry']
            else:
                doc_id_to_remove = None

            self.delete_doc_by_prop(remove_id, "_id", database, self.ANNOTATION_DATA_COLLECTION)

    def delete_doc_by_prop(self, vals: Union[list, str], property: str, database: str, collection):
        filter = []

        if not isinstance(vals, list):
            vals = [vals]

        for n in vals:
            filter.append({property: n})

        filter = {"$and": filter}

        return self.client[database][collection].delete_one(filter)

    def get_data_docs_by_prop(self, vals: Union[list, str], property: str, database: str):
        filter = []

        if not isinstance(vals, list):
            vals = [vals]

        for n in vals:
            filter.append({property: n})

        filter = {"$or": filter}

        result = list(self.client[database][self.ANNOTATION_DATA_COLLECTION].find(filter))[0]
        if "nextEntry" in result:
            return self.merge_collections(result, database)

        return result

    def get_fields_by_properties(self, vals: Union[list, str], property, fields: Union[list, str], database: str,
                                 collection: str):
        filter = []
        fields_dict = {}

        if not isinstance(fields, list):
            fields = [fields]

        for n in fields:
            fields_dict[n] = 1

        if not isinstance(vals, list):
            vals = [vals]

        for n in vals:
            filter.append({property: n})

        filter = {"$or": filter}

        return self.client[database][collection].find_one(filter, fields_dict)

    def merge_collections(self, doc, database):
        next_id = doc["nextEntry"]
        new_doc = self.get_data_docs_by_prop(next_id, "_id", database)
        doc['labels'] += new_doc['labels']

        return doc


if __name__ == "__main__":
    db_handler = NovaDBHandler("../../local/nova_db_test.cfg")

    test_cont = False
    test_cat = False
    test_free = True

    # Test continuous data download and upload
    if test_cont:
        dataset = "aria-noxi"
        session = "004_2016-03-18_Paris"
        scheme = "engagement"
        annotator = "system"
        roles = ["novice"]

        mongo_scheme = db_handler.get_schemes(dataset=dataset, schemes=[scheme])
        annos = db_handler.get_annos(
            dataset=dataset,
            scheme=scheme,
            session=session,
            annotator=annotator,
            roles=roles,
        )

    # Test categorical data download and upload
    if test_cat:
        dataset = "roxi"
        session = "001"
        scheme = "emotionalbursts"
        annotator = "gold"
        roles = ["player1"]

        mongo_scheme = db_handler.get_schemes(dataset=dataset, schemes=[scheme])
        annos = db_handler.get_annos(
            dataset=dataset,
            scheme=scheme,
            session=session,
            annotator=annotator,
            roles=roles,
        )

        new_annotator = "test"
        new_annos = [
            {"from": 0, "to": 10, "id": 1, "conf": 0.5},
            {"from": 20, "to": 25, "id": 1, "conf": 1},
            {"from": 30, "to": 35, "id": 1, "conf": 1},
        ]

        db_handler.set_annos(
            dataset=dataset,
            scheme=scheme,
            session=session,
            annotator=new_annotator,
            role=roles[0],
            annos=new_annos,
        )

    # Test free label download and upload
    if test_free:
        dataset = "kassel_therapie_korpus"
        session = "OPD_102"
        scheme = "transcript"
        annotator = "system"
        roles = ["therapist"]

        mongo_scheme = db_handler.get_schemes(dataset=dataset, schemes=[scheme])
        annos = db_handler.get_annos(
            dataset=dataset,
            scheme=scheme,
            session=session,
            annotator=annotator,
            roles=roles,
        )

        new_annotator = "schildom"
        new_annos = [
            {"from": 0, "to": 10, "conf": 1, "name": 'das'},
            {"from": 20, "to": 25, "conf": 1, "name": 'geht'},
            {"from": 30, "to": 35, "conf": 1, "name": 'ja'},
        ]

        db_handler.set_annos(
            dataset=dataset,
            scheme=scheme,
            session=session,
            annotator=new_annotator,
            role=roles[0],
            annos=new_annos,
        )

    print("Done")
