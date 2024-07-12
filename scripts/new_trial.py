
import datetime
import pytz

file_name = datetime.now(pytz.timezone('US/Mountain'))          # ~~~ current date and time MST
file_name = file_name[:file_name.find(".")].replace(" ","_")    # ~~~ remove the number of seconds (indicated with ".") and replace blank space (between date and time) with an underscore