import configparser
import os
import warnings
from datetime import datetime
from pymongo import MongoClient
from pymongo.results import InsertOneResult, UpdateResult
from typing import Union

ANNOTATOR_COLLECTION = "Annotators"
SCHEME_COLLECTION = "Schemes"
STREAM_COLLECTION = "Streams"
ROLE_COLLECTION = "Roles"
ANNOTATION_COLLECTION = "Annotations"
SESSION_COLLECTION = "Sessions"
ANNOTATION_DATA_COLLECTION = "AnnotationData"


class NovaDBHandler:
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
                scheme, "name", dataset, SCHEME_COLLECTION
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
            session, "name", dataset, SESSION_COLLECTION
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
                stream, "name", dataset, STREAM_COLLECTION
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
            scheme, "name", dataset, SCHEME_COLLECTION
        )
        if not mongo_schemes:
            warnings.warn(f"Unknown scheme {scheme} found")
            return []
        mongo_annotators = self.get_docs_by_prop(
            annotator, "name", dataset, ANNOTATOR_COLLECTION
        )
        if not mongo_annotators:
            warnings.warn(f"Unknown annotator {annotator} found")
            return []
        mongo_roles = self.get_docs_by_prop(roles, "name", dataset, ROLE_COLLECTION)
        if not mongo_roles:
            warnings.warn(f"Unknown role {roles} found")
            return []
        mongo_sessions = self.get_docs_by_prop(
            session, "name", dataset, SESSION_COLLECTION
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
            ANNOTATION_COLLECTION,
        )

        # getting the annotation data and the session name
        if not mongo_annos:
            print(
                f"No annotions found for \n\t-annotator: {annotator}\n\t-scheme: {scheme}\n\t-session: {session}\n\t-role: {roles}"
            )
            return []

        else:
            # TODO: adapt for multiple roles, annotators etc.
            label = self.get_docs_by_prop(
                mongo_annos[0]["data_id"], "_id", dataset, ANNOTATION_DATA_COLLECTION
            )
            label = label[0]["labels"]

        return label

    # Writing to database
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

    # TODO: Remove Restclass Labels in discrete Cases
    # TODO: Consider "forced overwrite"
    # TODO: Add Backup case
    # TODO: Call preprocess of annotation
    def set_annos(
        self,
        dataset: str,
        scheme: str,
        session: str,
        annotator: str,
        role: str,
        annos: list,
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
        mongo_scheme = self.get_docs_by_prop(scheme, "name", dataset, SCHEME_COLLECTION)
        if not mongo_scheme:
            warnings.warn(f"Unknown scheme {scheme} found")
            return ""

        mongo_annotator = self.get_docs_by_prop(
            annotator, "name", dataset, ANNOTATOR_COLLECTION
        )
        if not mongo_annotator:
            warnings.warn(f"Unknown annotator {annotator} found")
            return ""

        mongo_role = self.get_docs_by_prop(role, "name", dataset, ROLE_COLLECTION)
        if not mongo_role:
            warnings.warn(f"Unknown role {roles} found")
            return ""
        mongo_session = self.get_docs_by_prop(
            session, "name", dataset, SESSION_COLLECTION
        )
        if not mongo_session:
            warnings.warn(f"Unknown for session {session} found")
            return ""

        # Check if annotations already exist
        mongo_annos = self.get_annotation_docs(
            mongo_scheme,
            mongo_session,
            mongo_annotator,
            mongo_role,
            dataset,
            ANNOTATION_COLLECTION,
        )

        # Check for existing annotations
        mongo_anno_id = None
        mongo_data_id = None
        if mongo_annos:
            if mongo_annos[0]["isLocked"]:
                warnings.warn(
                    f"Can't overwrite locked annotation {str(mongo_annos[0]['_id'])}"
                )
                return ""
            else:
                warnings.warn(
                    f"Overwriting existing annotation {str(mongo_annos[0]['_id'])}"
                )
                mongo_anno_id = mongo_annos[0]["_id"]
                mongo_data_id = mongo_annos[0]["data_id"]

        # Upload label data
        mongo_label_doc = {"labels": annos}
        if mongo_data_id:
            mongo_label_doc["_id"] = mongo_data_id
            success = self.update_doc_by_prop(
                doc=mongo_label_doc,
                database=dataset,
                collection=ANNOTATION_DATA_COLLECTION,
            )
            if not success.acknowledged:
                warnings.warn(
                    f"Unknown error update database entries for Annotation data {mongo_data_id}"
                )
                return ""
            else:
                data_id = mongo_data_id
        else:
            success = self.insert_doc_by_prop(
                doc=mongo_label_doc,
                database=dataset,
                collection=ANNOTATION_DATA_COLLECTION,
            )
            if not success.acknowledged:
                warnings.warn(
                    f"Unexpected error uploading annotation data for {dataset} - {session} - {scheme} - {annotator}. Upload failed."
                )
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
            "isFinished": True,
            "isLocked": True,
            "date": datetime.today().replace(microsecond=0),
        }
        if mongo_anno_id:
            mongo_anno_doc["_id"] = mongo_anno_id
            success = self.update_doc_by_prop(
                doc=mongo_anno_doc, database=dataset, collection=ANNOTATION_COLLECTION
            )
            if not success.acknowledged:
                warnings.warn(
                    f"Unexpected error uploading annotations for {dataset} - {session} - {scheme} - {annotator}. "
                    f"Upload failed. "
                )
                return ""
            else:
                anno_id = mongo_anno_id
        else:
            success = self.insert_doc_by_prop(
                doc=mongo_anno_doc, database=dataset, collection=ANNOTATION_COLLECTION
            )
            if not success.acknowledged:
                warnings.warn(
                    f"Unexpected error uploading annotations for {dataset} - {session} - {scheme} - {annotator}. Upload failed."
                )
                return ""
            else:
                anno_id = success.inserted_id
        return anno_id


if __name__ == "__main__":
    db_handler = NovaDBHandler("../../local_data/nova_db_test.cfg")

    test_cont = False
    test_cat = True

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


    print("Done")
