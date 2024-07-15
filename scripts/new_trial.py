
import datetime
import pytz
from quality_of_life.my_base_utils import process_for_saving, dict_to_json

file_name = datetime.now(pytz.timezone('US/Mountain'))          # ~~~ current date and time MST
file_name = file_name[:file_name.find(".")].replace(" ","_")    # ~~~ remove the number of seconds (indicated with ".") and replace blank space (between date and time) with an underscore
file_name = process_for_saving(file_name+".json")               # ~~~ in case two trials were started within the same minute, e.g., procsess_for_saving("path_that_exists.json") returns "path_that_exists (1).json"
