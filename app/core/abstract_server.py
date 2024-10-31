import streamlit as st
from typing import Any

from app.core.system import AutoMLSystem
from app.core.streamlit_utils import GeneralUI


class AbstractServer:
    """
    This class serves as a parent class for all classes trying to
    view items in the database. Despite the name AbstractServer there
    is nothing abstract about this class.
    """
    def __init__(self, item_name: str):
        """
        A way of initialising an abstract server.

        :param item_name: the name of the item you wish to look for
        in the database
        :type item_name: str
        """
        self.ui_manager = GeneralUI()
        self._automl = AutoMLSystem.get_instance()
        self._items = self._automl.registry.list(type=item_name)
        self._item_name = item_name

    def _handle_view_saved_items_logic(self) -> dict:
        """
        Handle generic item viewing logic.
        """
        # Get a dictionary of the items and display them
        items_dict = self._items
        if not items_dict:
            self.ui_manager.display_error(f"No {self._item_name} available.")
            return
        
        # Get the name and the id of the items
        item_list, item_id_list = self.\
            _get_name_and_id_lists(items_dict)

        # Select an item
        selected_item = self.\
            _choose_from_saved_items(item_list)

        # Get the slected item's id
        selected_item_id = self.\
            _get_id(selected_item, item_list, item_id_list)

        # Get the attributes of the item
        if selected_item != f"Select a {self._item_name}":
            self.ui_manager.progress_bar()
            item_attributes = self.\
                _get_item_attributes(selected_item_id)
            return item_attributes

    def _display_item(self):
        """
        A way of displaying an item. Must impliment in children!
        """
        raise NotImplementedError("You need to implement this method")

    def _choose_from_saved_items(
            self,
            item_list: list[Any]
            ) -> str:
        """
        Display the list of saved items and allow the user to select one.

        :param item_list: the list of saved items
        :type item_list: list[Any]
        :return selcted_item: the choice of the user
        :type return: str
        """
        st.subheader(f"View Existing {self._item_name}:")
        selected_item = st.selectbox("Choose an item", item_list)
        return selected_item

    def _get_name_and_id_lists(
            self,
            item_dict: dict[Any]
            ) -> tuple[list[str], list[str]]:
        """
        Creates two list, one containing the names of the items
        that are avaliable along with their version. The other
        contains the ids of said lists.

        :param item_dict: the dictinary containing the items
        :type item_dict: dict[Any]
        :return: a list with names and a list with ids
        :rtype: tuple[list[str], list[str]]
        """
        item_id_list = ["No Id"]
        item_list = [f"Select a {self._item_name}"]
        for item in item_dict:
            name = item.name
            version = item.version 
            display_name = name + " " + "(version" + " " + version + ")"
            item_list.append(display_name)
            item_id_list.append(item.id)
        return item_list, item_id_list
    
    def _get_id(
            self,
            selected_item: str,
            item_list: list[str],
            item_id_list: list[str]
            ) -> str:
        """
        A way of getting the id of the slected item.

        :param selected_item: the selected item's name
        :type selected_item: str
        :param item_list: the list of avaliable items
        :type item_list: list[Any]
        :param item_id_list: the list of the ids of the avaliable
        items
        :type item_id_list: list[Any]
        :return: the id of the selected item
        :rtype: str
        """
        selected_id_index = item_list.index(selected_item)
        selected_item_id = item_id_list[selected_id_index]
        return selected_item_id
    
    def _get_item_attributes(
            self,
            selected_item_id: str
            ) -> Any:
        """
        Gets the attributes of the selected item.

        :param selected_item_id: the id of the selected item
        :type selected_item_id: str
        :return: the attributes of the selected item
        :rtype: Any
        """
        selected_artifact = self._automl.registry.get(selected_item_id)
        artifact_attributes = vars(selected_artifact)
        return artifact_attributes
    