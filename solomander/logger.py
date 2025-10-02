import logging
import inspect
import os

## =====================================
## Custom Logging Class
## =====================================


#log.SHOW will always be displayed, unless logging.CRITICAL is active
#log.CRITICAL will effectively turn off logging
# NOSET 0, DEBUG 10, INFO 20, WARNING 30, ERROR 40, CRITICAL 50, SHOW LEVEL 49
# Every level above is shown

SHOW_LEVEL = 49
SUCCESS_LEVEL = 35
INPUT_LEVEL = 48
logging.addLevelName(SHOW_LEVEL, "SHOW")
logging.addLevelName(SUCCESS_LEVEL, "SUCCESS")
logging.addLevelName(INPUT_LEVEL, "INPUT")
logging.SHOW = SHOW_LEVEL
logging.SUCCESS = SUCCESS_LEVEL
logging.INPUT = INPUT_LEVEL

log_level   = logging.DEBUG
stamp_level = logging.DEBUG
pront_level = logging.DEBUG


COLORS = {
        'blue': '\033[94m',    # Blue
        'green': '\033[92m',     # Green
        'neon_green': '\033[38;5;46m',
        'lime': '\033[38;5;118m',  # lime
        'yellow': '\033[93m',  # Yellow
        'red': '\033[91m',    # Red
        'bold_red': '\033[1;91m', # Bold Red
        'white': '\033[97m',  # White
        'magenta': '\033[95m',        # Magenta
        'underline_cyan': '\033[4;38;5;51m'
    }

RESET = '\033[0m'


class CustomLogger(logging.Logger):

    # custom calls for stamp.show()
    def show(self, msg, *args, **kwargs):
        if self.isEnabledFor(SHOW_LEVEL):
            self._log(SHOW_LEVEL, msg, args, stacklevel=2,**kwargs)
    
    # custom calls for stamp.sucess()
    def success(self, msg, *args, **kwargs):
        if self.isEnabledFor(SUCCESS_LEVEL):
            self._log(SUCCESS_LEVEL, msg, args, stacklevel=2,**kwargs)

    # custom calls for stamp.input()
    def input(self, msg, *args, **kwargs):
        if self.isEnabledFor(INPUT_LEVEL):
            self._log(INPUT_LEVEL, msg, args, stacklevel=2,**kwargs)

class CustomFormatter(logging.Formatter):

    LEVEL_COLORS = {
        logging.DEBUG: COLORS['magenta'],        #10
        logging.INFO: COLORS['blue'],            #20 
        logging.WARNING: COLORS['yellow'],       #30
        logging.SUCCESS: COLORS['lime'],         #35
        logging.ERROR: COLORS['red'],            #40
        logging.INPUT: COLORS['underline_cyan'], #48
        logging.SHOW: COLORS['white'],           #49
        logging.CRITICAL: COLORS['bold_red']     #50
    }

    def format(self, record):
        
        color = self.LEVEL_COLORS.get(record.levelno, COLORS['white']) # set color

        #custom time
        asctime = self.formatTime(record, self.datefmt)
        msecs = f"{int(record.msecs):03d}"
        record.color_time = f"{color}[{asctime}.{msecs}]{RESET}" # Color the entire timestamp including brackets and milliseconds

        #custom levelnames
        record.color_levelname = f"{color}[{record.levelname}]{RESET}"

        # custom filename, function name, line number
        record.color_file_lineo = f"{color}{record.filename}|{record.funcName}|ln{record.lineno}|{RESET}"

        #record.module_func_lineno = f"{record.module_func_lineno:<40}"
        record.color_message = f"{color}{record.getMessage()}{RESET}"
        
        return super().format(record)



## =====================================
## Loggers
## =====================================

## debugging ============
    
log = CustomLogger("solomander_logger")
log.setLevel(log_level)
log.propagate = False  # Prevent propagation to the root logger
log_console_handler = logging.StreamHandler()
log_console_handler.setLevel(log_level)
log_formatter = CustomFormatter('%(color_time)s %(color_levelname)s %(color_file_lineo)s %(color_message)-10s',datefmt='%H:%M:%S')
log_console_handler.setFormatter(log_formatter)
log.addHandler(log_console_handler)

## time stamping ============

stamp = CustomLogger("solomander_stamper")
stamp.setLevel(stamp_level)
stamp.propagate = False 
stamp_console_handler = logging.StreamHandler()
stamp_console_handler.setLevel(stamp_level)
stamp_formatter = CustomFormatter('%(color_time)s %(color_message)s', datefmt='%H:%M:%S')
stamp_console_handler.setFormatter(stamp_formatter)
stamp.addHandler(stamp_console_handler)

## printing ============

pront = CustomLogger("solomander_printer")
pront.setLevel(pront_level)
pront.propagate = False 
print_console_handler = logging.StreamHandler()
print_console_handler.setLevel(pront_level)
print_formatter = CustomFormatter('%(color_message)s')
print_console_handler.setFormatter(print_formatter)
pront.addHandler(print_console_handler)


def set_log_level(all=None, log_level=None, stamp_level=None, pront_level=None):
    
    if all is not None:
        log.setLevel(all)
        stamp.setLevel(all)
        pront.setLevel(all)
        return
    
    if log_level or all is not None:
        log.setLevel(log_level)
    if stamp_level is not None:
        stamp.setLevel(stamp_level)
    if pront_level is not None:
        pront.setLevel(pront_level)


if __name__ == '__main__':
    
    set_log_level(log_level=10)

    log.debug("debug")
    log.info("info")
    log.warning("warning")
    log.error("error")
    log.critical("critical")
    stamp.info("info")
    stamp.show("hello")
    stamp.success("success")
   
    
    