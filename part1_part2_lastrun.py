#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2024.2.4),
    on Juni 23, 2025, at 13:26
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '3'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2024.2.4'
expName = 'part1_part2'  # from the Builder filename that created this script
# information about this experiment
expInfo = {
    'PID': '',
    'Birthyear': '',
    'Gender': '',
    'Handedness': '',
    'MRI': '1',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = [1024, 768]
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/Guess_%s_%s_%s' % (expInfo['PID'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='C:\\Users\\Nemecz\\Documents\\Guess_fMRI\\part1_part2_lastrun.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('info')
        )
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowGUI=False, allowStencil=False,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win._monitorFrameRate = win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.hideMessage()
    # show a visual indicator if we're in piloting mode
    if PILOTING and prefs.piloting['showPilotingIndicator']:
        win.showPilotingIndicator()
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    # Setup iohub experiment
    ioConfig['Experiment'] = dict(filename=thisExp.dataFileName)
    
    # Start ioHub server
    ioServer = io.launchHubServer(window=win, **ioConfig)
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='iohub'
        )
    if deviceManager.getDevice('start_experiment_key') is None:
        # initialise start_experiment_key
        start_experiment_key = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='start_experiment_key',
        )
    if deviceManager.getDevice('instruction1_key') is None:
        # initialise instruction1_key
        instruction1_key = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='instruction1_key',
        )
    if deviceManager.getDevice('instruction2_key') is None:
        # initialise instruction2_key
        instruction2_key = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='instruction2_key',
        )
    if deviceManager.getDevice('instruction3_key') is None:
        # initialise instruction3_key
        instruction3_key = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='instruction3_key',
        )
    if deviceManager.getDevice('instruction4_key') is None:
        # initialise instruction4_key
        instruction4_key = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='instruction4_key',
        )
    if deviceManager.getDevice('instruction5_key') is None:
        # initialise instruction5_key
        instruction5_key = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='instruction5_key',
        )
    if deviceManager.getDevice('guess_reached') is None:
        # initialise guess_reached
        guess_reached = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='guess_reached',
        )
    if deviceManager.getDevice('end_encoding') is None:
        # initialise end_encoding
        end_encoding = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='end_encoding',
        )
    if deviceManager.getDevice('start_key') is None:
        # initialise start_key
        start_key = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='start_key',
        )
    if deviceManager.getDevice('scanner_ready_press') is None:
        # initialise scanner_ready_press
        scanner_ready_press = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='scanner_ready_press',
        )
    if deviceManager.getDevice('skip_trigger') is None:
        # initialise skip_trigger
        skip_trigger = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='skip_trigger',
        )
    # create speaker 'sound_1'
    deviceManager.addDevice(
        deviceName='sound_1',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index=-1
    )
    # create speaker 'sound_2'
    deviceManager.addDevice(
        deviceName='sound_2',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index=-1
    )
    if deviceManager.getDevice('task_break_resp') is None:
        # initialise task_break_resp
        task_break_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='task_break_resp',
        )
    if deviceManager.getDevice('end_part1_key') is None:
        # initialise end_part1_key
        end_part1_key = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='end_part1_key',
        )
    if deviceManager.getDevice('key_resp_dist_instr') is None:
        # initialise key_resp_dist_instr
        key_resp_dist_instr = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_dist_instr',
        )
    if deviceManager.getDevice('key_resp_gaze') is None:
        # initialise key_resp_gaze
        key_resp_gaze = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_gaze',
        )
    if deviceManager.getDevice('recall_instructions1_key') is None:
        # initialise recall_instructions1_key
        recall_instructions1_key = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='recall_instructions1_key',
        )
    if deviceManager.getDevice('recall_instructions2_key') is None:
        # initialise recall_instructions2_key
        recall_instructions2_key = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='recall_instructions2_key',
        )
    if deviceManager.getDevice('end_cued_recall') is None:
        # initialise end_cued_recall
        end_cued_recall = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='end_cued_recall',
        )
    if deviceManager.getDevice('recall_reached') is None:
        # initialise recall_reached
        recall_reached = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='recall_reached',
        )
    if deviceManager.getDevice('recall_selection') is None:
        # initialise recall_selection
        recall_selection = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='recall_selection',
        )
    if deviceManager.getDevice('end_feedback') is None:
        # initialise end_feedback
        end_feedback = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='end_feedback',
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='ioHub',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure window is set to foreground to prevent losing focus
    win.winHandle.activate()
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ioHub'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "welcome" ---
    welcome_text = visual.TextStim(win=win, name='welcome_text',
        text='Willkommen zum Ratespiel Experiment!',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    # Run 'Begin Experiment' code from experiment_setup
    # import random for letter randomization
    import random as rnd
    
    if expInfo["MRI"] == "1":
        from psychopy import parallel
        port = parallel.ParallelPort(address = 0x2FE8) 
        pinNumber = 10 #Change to match the pin that is receiving the pulse value sent by your scanner. Set this to None to scan all pins
    
    all_keys = ['space', '1', '2', '3', '4', '5']
    guess_made_key = "2"
    no_guess_made_key = "3"
    
    guess_dur = 3
    guess_resp_dur = 3
    guess_delay_dur = 3
    
    cue_types = {}
    target_types = {}
    
    succ_recall_key = "3"
    letter_dur = 3
    
    run_counter = 1
    
    ncorrect = 0
    correct_guess = 0
    correct_read = 0
    
    num_trials = 40 # number of trials per run
    if expInfo["PID"] == "pilot" :
        num_trials = 2
    
    start_experiment_key = keyboard.Keyboard(deviceName='start_experiment_key')
    
    # --- Initialize components for Routine "instructions1" ---
    instructions1_text = visual.TextStim(win=win, name='instructions1_text',
        text='Teil 1\n\nIm ersten Teil des Experiments werden Ihnen Wortpaare auf dem Bildschirm angezeigt.\n\nZum Beispiel:\n\nComputer - Tastatur\n\nIhre Aufgabe ist es, sich so viele Wortpaare wie möglich einzuprägen.\n\nDrücken Sie eine Taste, um fortzufahren.',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=1.2, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    instruction1_key = keyboard.Keyboard(deviceName='instruction1_key')
    
    # --- Initialize components for Routine "instructions2" ---
    instructions2_text = visual.TextStim(win=win, name='instructions2_text',
        text='Manchmal wird Ihnen ein einzelnes Wort auf dem Bildschirm angezeigt, bevor Sie das gesamte Wortpaar sehen, und Sie müssen raten, was das andere Wort sein könnte.\n\nZum Beispiel:\n\nComputer - ?\n\nDrücken Sie eine Taste, um fortzufahren.',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=1.2, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    instruction2_key = keyboard.Keyboard(deviceName='instruction2_key')
    
    # --- Initialize components for Routine "instructions3" ---
    instructions3_text = visual.TextStim(win=win, name='instructions3_text',
        text='Raten Sie nicht laut, sondern versuchen Sie möglichst viele Assoziationen zum linken Wort leise in Ihrem Kopf zu generieren.\n\nZum Beispiel könnten Ihnen zum Wort "Sommerferien" folgende Assoziationen einfallen: Urlaub, Strand, Reisen, Ferienjob, Eis und so weiter.\n\nDrücken Sie eine Taste, um fortzufahren.',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=1.2, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    instruction3_key = keyboard.Keyboard(deviceName='instruction3_key')
    
    # --- Initialize components for Routine "instructions4" ---
    instructions4_text = visual.TextStim(win=win, name='instructions4_text',
        text='Sie haben 3 Sekunden Zeit, um nach dem Erscheinen des linken Wortes mindestens einen stillen Rateversuch zu machen. Sie müssen angeben, ob Sie einen Rateversuch machen konnten, indem Sie die Tasten 1 oder 2 drücken.\n\n1) Rateversuch generiert 2) Keinen Rateversuch generiert\n\nDrücken Sie eine Taste, um fortzufahren.',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=1.2, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    instruction4_key = keyboard.Keyboard(deviceName='instruction4_key')
    
    # --- Initialize components for Routine "instructions5" ---
    instructions5_text = visual.TextStim(win=win, name='instructions5_text',
        text='Zusammenfassung:\nIhre Aufgabe ist es, so viele der auf dem Bildschirm präsentierten Wortpaare wie möglich zu lernen.\n\nManchmal wird Ihnen zunächst nur ein einzelnes Wort angezeigt, und Sie müssen raten, was das zugehörige Wortpaar sein könnte. Drücken Sie 1 oder 2, um anzugeben, ob Sie einen Rateversuch gemacht haben.\n\nVor dem Experiment werden Sie die Aufgabe üben.\n\nDrücken Sie eine Taste, um mit der Übung zu beginnen.\n',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=1.2, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    instruction5_key = keyboard.Keyboard(deviceName='instruction5_key')
    
    # --- Initialize components for Routine "iti_learning_practice" ---
    iti_practice_blank = visual.TextStim(win=win, name='iti_practice_blank',
        text='+',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "guess_response" ---
    dash_stim_guessing_resp = visual.TextStim(win=win, name='dash_stim_guessing_resp',
        text='-',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    cue_stim_guessing_resp = visual.TextStim(win=win, name='cue_stim_guessing_resp',
        text='',
        font='Arial',
        pos=(-0.2, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    guess_stim_resp = visual.TextStim(win=win, name='guess_stim_resp',
        text='?',
        font='Arial',
        pos=(0.2, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    guess_question = visual.TextStim(win=win, name='guess_question',
        text='\n\n1) Ja        2) Nein',
        font='Arial',
        pos=(0, -0.3), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    guess_reached = keyboard.Keyboard(deviceName='guess_reached')
    
    # --- Initialize components for Routine "feedback_guess" ---
    no_response_feedback = visual.TextStim(win=win, name='no_response_feedback',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "guess_delay" ---
    guess_delay_cross = visual.TextStim(win=win, name='guess_delay_cross',
        text='+',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "encode" ---
    dash_stim_learning = visual.TextStim(win=win, name='dash_stim_learning',
        text='-',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    cue_stim_learning = visual.TextStim(win=win, name='cue_stim_learning',
        text='',
        font='Arial',
        pos=(-0.2, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    target_stim = visual.TextStim(win=win, name='target_stim',
        text='',
        font='Arial',
        pos=(0.2, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    end_encoding = keyboard.Keyboard(deviceName='end_encoding')
    
    # --- Initialize components for Routine "load_word_pairs" ---
    # Run 'Begin Experiment' code from load_stimuli
    cue_list = []
    target_list = []
    
    
    # --- Initialize components for Routine "start_task" ---
    start_task_text = visual.TextStim(win=win, name='start_task_text',
        text='Jetzt beginnt die Aufgabe. Sie erhalten keine Rückmeldungen mehr.\n\nDrücken Sie eine Taste, wenn Sie bereit sind zu starten.',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    start_key = keyboard.Keyboard(deviceName='start_key')
    
    # --- Initialize components for Routine "load_guess_conditions" ---
    # Run 'Begin Experiment' code from load_guess_list
    guess_list = []
    practice_list = ["Read", "Guess", "Guess", "Read"]
    
    
    # --- Initialize components for Routine "load_iti" ---
    # Run 'Begin Experiment' code from load_iti_code
    iti_list = []
    
    
    # --- Initialize components for Routine "load_delay" ---
    # Run 'Begin Experiment' code from load_delay_code
    delay_list = []
    
    
    # --- Initialize components for Routine "prep_scanner" ---
    scanner_ready_press = keyboard.Keyboard(deviceName='scanner_ready_press')
    ready_set_text = visual.TextStim(win=win, name='ready_set_text',
        text='Der Scanner wird vorbereitet.',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "wait_for_trigger" ---
    starting_soon = visual.TextStim(win=win, name='starting_soon',
        text='fMRT startet gleich...',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    skip_trigger = keyboard.Keyboard(deviceName='skip_trigger')
    
    # --- Initialize components for Routine "blank" ---
    begin_end_run_cross = visual.TextStim(win=win, name='begin_end_run_cross',
        text='+',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    sound_1 = sound.Sound(
        'A', 
        secs=1.2, 
        stereo=True, 
        hamming=True, 
        speaker='sound_1',    name='sound_1'
    )
    sound_1.setVolume(1.0)
    sound_2 = sound.Sound(
        'A', 
        secs=1, 
        stereo=True, 
        hamming=True, 
        speaker='sound_2',    name='sound_2'
    )
    sound_2.setVolume(1.0)
    
    # --- Initialize components for Routine "setup_learning_trial" ---
    
    # --- Initialize components for Routine "guess_response" ---
    dash_stim_guessing_resp = visual.TextStim(win=win, name='dash_stim_guessing_resp',
        text='-',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    cue_stim_guessing_resp = visual.TextStim(win=win, name='cue_stim_guessing_resp',
        text='',
        font='Arial',
        pos=(-0.2, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    guess_stim_resp = visual.TextStim(win=win, name='guess_stim_resp',
        text='?',
        font='Arial',
        pos=(0.2, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    guess_question = visual.TextStim(win=win, name='guess_question',
        text='\n\n1) Ja        2) Nein',
        font='Arial',
        pos=(0, -0.3), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    guess_reached = keyboard.Keyboard(deviceName='guess_reached')
    
    # --- Initialize components for Routine "guess_delay" ---
    guess_delay_cross = visual.TextStim(win=win, name='guess_delay_cross',
        text='+',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "encode" ---
    dash_stim_learning = visual.TextStim(win=win, name='dash_stim_learning',
        text='-',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    cue_stim_learning = visual.TextStim(win=win, name='cue_stim_learning',
        text='',
        font='Arial',
        pos=(-0.2, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    target_stim = visual.TextStim(win=win, name='target_stim',
        text='',
        font='Arial',
        pos=(0.2, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    end_encoding = keyboard.Keyboard(deviceName='end_encoding')
    
    # --- Initialize components for Routine "iti_task" ---
    iti_cross = visual.TextStim(win=win, name='iti_cross',
        text='+',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "blank" ---
    begin_end_run_cross = visual.TextStim(win=win, name='begin_end_run_cross',
        text='+',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    sound_1 = sound.Sound(
        'A', 
        secs=1.2, 
        stereo=True, 
        hamming=True, 
        speaker='sound_1',    name='sound_1'
    )
    sound_1.setVolume(1.0)
    sound_2 = sound.Sound(
        'A', 
        secs=1, 
        stereo=True, 
        hamming=True, 
        speaker='sound_2',    name='sound_2'
    )
    sound_2.setVolume(1.0)
    
    # --- Initialize components for Routine "task_break" ---
    task_break_text = visual.TextStim(win=win, name='task_break_text',
        text='Pause. Drücken Sie eine Taste, um fortzufahren.',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    task_break_resp = keyboard.Keyboard(deviceName='task_break_resp')
    
    # --- Initialize components for Routine "end_part1" ---
    end_part1_key = keyboard.Keyboard(deviceName='end_part1_key')
    end_part1_text = visual.TextStim(win=win, name='end_part1_text',
        text='Sie haben den ersten Teil des Experiments abgeschlossen!\n\nDrücken Sie eine Taste, um fortzufahren.',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "distractor_instruction" ---
    dist_instruc = visual.TextStim(win=win, name='dist_instruc',
        text='Es folgt nun eine kurze Aufgabe.\n\nIn der Mitte des Bildschirms erscheint ein Smiley, der nach rechts und links schaut. Neben ihm erscheint ein grüner und roter Kreis.\n\nIhre Aufgabe ist es anzugeben, wo der GRÜNE Kreis auftaucht.\nWenn er LINKS auftaucht drücken Sie TASTE 1, wenn er RECHTS auftaucht drücken Sie TASTE 2. \n\nStarten Sie die Übung durch das Drücken einer beliebigen Taste.',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_dist_instr = keyboard.Keyboard(deviceName='key_resp_dist_instr')
    
    # --- Initialize components for Routine "start_dist_prac" ---
    start_dist_prac_text = visual.TextStim(win=win, name='start_dist_prac_text',
        text='Jetzt beginnt die Übung. ',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "dist_trial" ---
    Smileycentral = visual.ImageStim(
        win=win,
        name='Smileycentral', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(1, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    key_resp_gaze = keyboard.Keyboard(deviceName='key_resp_gaze')
    Smiley = visual.ImageStim(
        win=win,
        name='Smiley', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(1,0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    circle = visual.ShapeStim(
        win=win, name='circle',
        size=(0.05, 0.05), vertices='circle',
        ori=0.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=[-1.0000, -0.2157, -1.0000], fillColor=[-1.0000, -0.2157, -1.0000],
        opacity=None, depth=-3.0, interpolate=True)
    distractorcircle = visual.ShapeStim(
        win=win, name='distractorcircle',
        size=(0.05, 0.05), vertices='circle',
        ori=0.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=[0.7255, -0.8431, -0.5294], fillColor=[0.7255, -0.8431, -0.5294],
        opacity=None, depth=-4.0, interpolate=True)
    # Run 'Begin Experiment' code from dist_code
    fback_text = ""
    rt = ""
    dt = 0
    
    # --- Initialize components for Routine "dist_feedback" ---
    dist_fback_text = visual.TextStim(win=win, name='dist_fback_text',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "start_dist" ---
    start_dist_text = visual.TextStim(win=win, name='start_dist_text',
        text='Jetzt beginnt die Aufgabe. ',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "dist_trial" ---
    Smileycentral = visual.ImageStim(
        win=win,
        name='Smileycentral', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(1, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    key_resp_gaze = keyboard.Keyboard(deviceName='key_resp_gaze')
    Smiley = visual.ImageStim(
        win=win,
        name='Smiley', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(1,0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    circle = visual.ShapeStim(
        win=win, name='circle',
        size=(0.05, 0.05), vertices='circle',
        ori=0.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=[-1.0000, -0.2157, -1.0000], fillColor=[-1.0000, -0.2157, -1.0000],
        opacity=None, depth=-3.0, interpolate=True)
    distractorcircle = visual.ShapeStim(
        win=win, name='distractorcircle',
        size=(0.05, 0.05), vertices='circle',
        ori=0.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=[0.7255, -0.8431, -0.5294], fillColor=[0.7255, -0.8431, -0.5294],
        opacity=None, depth=-4.0, interpolate=True)
    # Run 'Begin Experiment' code from dist_code
    fback_text = ""
    rt = ""
    dt = 0
    
    # --- Initialize components for Routine "instructions_recall1" ---
    recall_instructions1_text = visual.TextStim(win=win, name='recall_instructions1_text',
        text='Im nächsten Teil des Experiments müssen Sie das Wortpaar zu jedem auf dem Bildschirm angezeigten Wort abrufen.\n\nVersuchen Sie, das Wort (3 Sekunden) abzurufen, und geben Sie dann an, ob Sie sich an das Wort erinnern, indem Sie die Tasten 1 (Kann mich nicht erinnern) und 2 (Kann mich erinnern) verwenden.\n\nDrücken Sie eine Taste, um fortzufahren.',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=1.2, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    recall_instructions1_key = keyboard.Keyboard(deviceName='recall_instructions1_key')
    
    # --- Initialize components for Routine "instructions_recall2" ---
    recall_instructions2_text = visual.TextStim(win=win, name='recall_instructions2_text',
        text='Wenn Sie sich an das Wort erinnern können, fragen wir Sie nach dem letzten Buchstaben des Wortes. Sie können aus 4 Optionen wählen (Tasten 1-4).\n\nBevor Sie starten, üben Sie die Aufgabe.\n\nDrücken Sie eine Taste, um mit der Übung zu beginnen.',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=1.2, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    recall_instructions2_key = keyboard.Keyboard(deviceName='recall_instructions2_key')
    
    # --- Initialize components for Routine "iti_recall_practice" ---
    iti_recall_practice_cross = visual.TextStim(win=win, name='iti_recall_practice_cross',
        text='+',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "cued_recall" ---
    dash_stim_recall = visual.TextStim(win=win, name='dash_stim_recall',
        text='-',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    qmark_target = visual.TextStim(win=win, name='qmark_target',
        text='?',
        font='Arial',
        pos=(0.2, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    end_cued_recall = keyboard.Keyboard(deviceName='end_cued_recall')
    cue_stim_recall = visual.TextStim(win=win, name='cue_stim_recall',
        text='',
        font='Arial',
        pos=(-0.2, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    
    # --- Initialize components for Routine "recall_response" ---
    dash_stim_recall_resp = visual.TextStim(win=win, name='dash_stim_recall_resp',
        text='-',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    cue_stim_recall_resp = visual.TextStim(win=win, name='cue_stim_recall_resp',
        text='',
        font='Arial',
        pos=(-0.2, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    recall_stim_resp = visual.TextStim(win=win, name='recall_stim_resp',
        text='?',
        font='Arial',
        pos=(0.2, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    recall_question = visual.TextStim(win=win, name='recall_question',
        text='1) Nein        2) Ja',
        font='Arial',
        pos=(0, -0.3), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    recall_reached = keyboard.Keyboard(deviceName='recall_reached')
    
    # --- Initialize components for Routine "recall_select" ---
    dash_stim_recall_select = visual.TextStim(win=win, name='dash_stim_recall_select',
        text='-',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    cue_stim_recall_select = visual.TextStim(win=win, name='cue_stim_recall_select',
        text='',
        font='Arial',
        pos=(-0.2, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    recall_stim_select = visual.TextStim(win=win, name='recall_stim_select',
        text='?',
        font='Arial',
        pos=(0.2, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    recall_question_select = visual.TextStim(win=win, name='recall_question_select',
        text='',
        font='Arial',
        pos=(0, -0.3), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    recall_selection = keyboard.Keyboard(deviceName='recall_selection')
    
    # --- Initialize components for Routine "recall_feedback" ---
    dash_stim_feedback = visual.TextStim(win=win, name='dash_stim_feedback',
        text='-',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    cue_stim_feedback = visual.TextStim(win=win, name='cue_stim_feedback',
        text='',
        font='Arial',
        pos=(-0.2, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    target_stim_feedback = visual.TextStim(win=win, name='target_stim_feedback',
        text='',
        font='Arial',
        pos=(0.2, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    end_feedback = keyboard.Keyboard(deviceName='end_feedback')
    recall_feedback1 = visual.TextStim(win=win, name='recall_feedback1',
        text='Der letzte Buchstabe:',
        font='Arial',
        pos=(-0.3, -0.3), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    recall_feedback2 = visual.TextStim(win=win, name='recall_feedback2',
        text='',
        font='Arial',
        pos=(0.0, -0.3), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-5.0);
    
    # --- Initialize components for Routine "load_word_pairs" ---
    # Run 'Begin Experiment' code from load_stimuli
    cue_list = []
    target_list = []
    
    
    # --- Initialize components for Routine "start_task" ---
    start_task_text = visual.TextStim(win=win, name='start_task_text',
        text='Jetzt beginnt die Aufgabe. Sie erhalten keine Rückmeldungen mehr.\n\nDrücken Sie eine Taste, wenn Sie bereit sind zu starten.',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    start_key = keyboard.Keyboard(deviceName='start_key')
    
    # --- Initialize components for Routine "load_iti" ---
    # Run 'Begin Experiment' code from load_iti_code
    iti_list = []
    
    
    # --- Initialize components for Routine "prep_scanner" ---
    scanner_ready_press = keyboard.Keyboard(deviceName='scanner_ready_press')
    ready_set_text = visual.TextStim(win=win, name='ready_set_text',
        text='Der Scanner wird vorbereitet.',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "wait_for_trigger" ---
    starting_soon = visual.TextStim(win=win, name='starting_soon',
        text='fMRT startet gleich...',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    skip_trigger = keyboard.Keyboard(deviceName='skip_trigger')
    
    # --- Initialize components for Routine "blank" ---
    begin_end_run_cross = visual.TextStim(win=win, name='begin_end_run_cross',
        text='+',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    sound_1 = sound.Sound(
        'A', 
        secs=1.2, 
        stereo=True, 
        hamming=True, 
        speaker='sound_1',    name='sound_1'
    )
    sound_1.setVolume(1.0)
    sound_2 = sound.Sound(
        'A', 
        secs=1, 
        stereo=True, 
        hamming=True, 
        speaker='sound_2',    name='sound_2'
    )
    sound_2.setVolume(1.0)
    
    # --- Initialize components for Routine "setup_recall_trial" ---
    
    # --- Initialize components for Routine "cued_recall" ---
    dash_stim_recall = visual.TextStim(win=win, name='dash_stim_recall',
        text='-',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    qmark_target = visual.TextStim(win=win, name='qmark_target',
        text='?',
        font='Arial',
        pos=(0.2, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    end_cued_recall = keyboard.Keyboard(deviceName='end_cued_recall')
    cue_stim_recall = visual.TextStim(win=win, name='cue_stim_recall',
        text='',
        font='Arial',
        pos=(-0.2, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    
    # --- Initialize components for Routine "recall_response" ---
    dash_stim_recall_resp = visual.TextStim(win=win, name='dash_stim_recall_resp',
        text='-',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    cue_stim_recall_resp = visual.TextStim(win=win, name='cue_stim_recall_resp',
        text='',
        font='Arial',
        pos=(-0.2, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    recall_stim_resp = visual.TextStim(win=win, name='recall_stim_resp',
        text='?',
        font='Arial',
        pos=(0.2, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    recall_question = visual.TextStim(win=win, name='recall_question',
        text='1) Nein        2) Ja',
        font='Arial',
        pos=(0, -0.3), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    recall_reached = keyboard.Keyboard(deviceName='recall_reached')
    
    # --- Initialize components for Routine "recall_select" ---
    dash_stim_recall_select = visual.TextStim(win=win, name='dash_stim_recall_select',
        text='-',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    cue_stim_recall_select = visual.TextStim(win=win, name='cue_stim_recall_select',
        text='',
        font='Arial',
        pos=(-0.2, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    recall_stim_select = visual.TextStim(win=win, name='recall_stim_select',
        text='?',
        font='Arial',
        pos=(0.2, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    recall_question_select = visual.TextStim(win=win, name='recall_question_select',
        text='',
        font='Arial',
        pos=(0, -0.3), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    recall_selection = keyboard.Keyboard(deviceName='recall_selection')
    
    # --- Initialize components for Routine "iti_task" ---
    iti_cross = visual.TextStim(win=win, name='iti_cross',
        text='+',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "blank" ---
    begin_end_run_cross = visual.TextStim(win=win, name='begin_end_run_cross',
        text='+',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    sound_1 = sound.Sound(
        'A', 
        secs=1.2, 
        stereo=True, 
        hamming=True, 
        speaker='sound_1',    name='sound_1'
    )
    sound_1.setVolume(1.0)
    sound_2 = sound.Sound(
        'A', 
        secs=1, 
        stereo=True, 
        hamming=True, 
        speaker='sound_2',    name='sound_2'
    )
    sound_2.setVolume(1.0)
    
    # --- Initialize components for Routine "task_break" ---
    task_break_text = visual.TextStim(win=win, name='task_break_text',
        text='Pause. Drücken Sie eine Taste, um fortzufahren.',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    task_break_resp = keyboard.Keyboard(deviceName='task_break_resp')
    
    # --- Initialize components for Routine "end_part2" ---
    end_part2_text = visual.TextStim(win=win, name='end_part2_text',
        text='Sie haben den zweiten Teil des Experiments abgeschlossen!',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "welcome" ---
    # create an object to store info about Routine welcome
    welcome = data.Routine(
        name='welcome',
        components=[welcome_text, start_experiment_key],
    )
    welcome.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for start_experiment_key
    start_experiment_key.keys = []
    start_experiment_key.rt = []
    _start_experiment_key_allKeys = []
    # allowedKeys looks like a variable, so make sure it exists locally
    if 'all_keys' in globals():
        all_keys = globals()['all_keys']
    # store start times for welcome
    welcome.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    welcome.tStart = globalClock.getTime(format='float')
    welcome.status = STARTED
    thisExp.addData('welcome.started', welcome.tStart)
    welcome.maxDuration = None
    # keep track of which components have finished
    welcomeComponents = welcome.components
    for thisComponent in welcome.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "welcome" ---
    welcome.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *welcome_text* updates
        
        # if welcome_text is starting this frame...
        if welcome_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            welcome_text.frameNStart = frameN  # exact frame index
            welcome_text.tStart = t  # local t and not account for scr refresh
            welcome_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(welcome_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'welcome_text.started')
            # update status
            welcome_text.status = STARTED
            welcome_text.setAutoDraw(True)
        
        # if welcome_text is active this frame...
        if welcome_text.status == STARTED:
            # update params
            pass
        
        # *start_experiment_key* updates
        waitOnFlip = False
        
        # if start_experiment_key is starting this frame...
        if start_experiment_key.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            start_experiment_key.frameNStart = frameN  # exact frame index
            start_experiment_key.tStart = t  # local t and not account for scr refresh
            start_experiment_key.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(start_experiment_key, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'start_experiment_key.started')
            # update status
            start_experiment_key.status = STARTED
            # allowed keys looks like a variable named `all_keys`
            if not type(all_keys) in [list, tuple, np.ndarray]:
                if not isinstance(all_keys, str):
                    all_keys = str(all_keys)
                elif not ',' in all_keys:
                    all_keys = (all_keys,)
                else:
                    all_keys = eval(all_keys)
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(start_experiment_key.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(start_experiment_key.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if start_experiment_key.status == STARTED and not waitOnFlip:
            theseKeys = start_experiment_key.getKeys(keyList=list(all_keys), ignoreKeys=["escape"], waitRelease=False)
            _start_experiment_key_allKeys.extend(theseKeys)
            if len(_start_experiment_key_allKeys):
                start_experiment_key.keys = _start_experiment_key_allKeys[-1].name  # just the last key pressed
                start_experiment_key.rt = _start_experiment_key_allKeys[-1].rt
                start_experiment_key.duration = _start_experiment_key_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            welcome.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in welcome.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "welcome" ---
    for thisComponent in welcome.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for welcome
    welcome.tStop = globalClock.getTime(format='float')
    welcome.tStopRefresh = tThisFlipGlobal
    thisExp.addData('welcome.stopped', welcome.tStop)
    # check responses
    if start_experiment_key.keys in ['', [], None]:  # No response was made
        start_experiment_key.keys = None
    thisExp.addData('start_experiment_key.keys',start_experiment_key.keys)
    if start_experiment_key.keys != None:  # we had a response
        thisExp.addData('start_experiment_key.rt', start_experiment_key.rt)
        thisExp.addData('start_experiment_key.duration', start_experiment_key.duration)
    thisExp.nextEntry()
    # the Routine "welcome" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "instructions1" ---
    # create an object to store info about Routine instructions1
    instructions1 = data.Routine(
        name='instructions1',
        components=[instructions1_text, instruction1_key],
    )
    instructions1.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for instruction1_key
    instruction1_key.keys = []
    instruction1_key.rt = []
    _instruction1_key_allKeys = []
    # allowedKeys looks like a variable, so make sure it exists locally
    if 'all_keys' in globals():
        all_keys = globals()['all_keys']
    # store start times for instructions1
    instructions1.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instructions1.tStart = globalClock.getTime(format='float')
    instructions1.status = STARTED
    thisExp.addData('instructions1.started', instructions1.tStart)
    instructions1.maxDuration = None
    # keep track of which components have finished
    instructions1Components = instructions1.components
    for thisComponent in instructions1.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instructions1" ---
    instructions1.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *instructions1_text* updates
        
        # if instructions1_text is starting this frame...
        if instructions1_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instructions1_text.frameNStart = frameN  # exact frame index
            instructions1_text.tStart = t  # local t and not account for scr refresh
            instructions1_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instructions1_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instructions1_text.started')
            # update status
            instructions1_text.status = STARTED
            instructions1_text.setAutoDraw(True)
        
        # if instructions1_text is active this frame...
        if instructions1_text.status == STARTED:
            # update params
            pass
        
        # *instruction1_key* updates
        waitOnFlip = False
        
        # if instruction1_key is starting this frame...
        if instruction1_key.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instruction1_key.frameNStart = frameN  # exact frame index
            instruction1_key.tStart = t  # local t and not account for scr refresh
            instruction1_key.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instruction1_key, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instruction1_key.started')
            # update status
            instruction1_key.status = STARTED
            # allowed keys looks like a variable named `all_keys`
            if not type(all_keys) in [list, tuple, np.ndarray]:
                if not isinstance(all_keys, str):
                    all_keys = str(all_keys)
                elif not ',' in all_keys:
                    all_keys = (all_keys,)
                else:
                    all_keys = eval(all_keys)
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(instruction1_key.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(instruction1_key.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if instruction1_key.status == STARTED and not waitOnFlip:
            theseKeys = instruction1_key.getKeys(keyList=list(all_keys), ignoreKeys=["escape"], waitRelease=False)
            _instruction1_key_allKeys.extend(theseKeys)
            if len(_instruction1_key_allKeys):
                instruction1_key.keys = _instruction1_key_allKeys[-1].name  # just the last key pressed
                instruction1_key.rt = _instruction1_key_allKeys[-1].rt
                instruction1_key.duration = _instruction1_key_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            instructions1.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instructions1.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instructions1" ---
    for thisComponent in instructions1.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instructions1
    instructions1.tStop = globalClock.getTime(format='float')
    instructions1.tStopRefresh = tThisFlipGlobal
    thisExp.addData('instructions1.stopped', instructions1.tStop)
    # check responses
    if instruction1_key.keys in ['', [], None]:  # No response was made
        instruction1_key.keys = None
    thisExp.addData('instruction1_key.keys',instruction1_key.keys)
    if instruction1_key.keys != None:  # we had a response
        thisExp.addData('instruction1_key.rt', instruction1_key.rt)
        thisExp.addData('instruction1_key.duration', instruction1_key.duration)
    thisExp.nextEntry()
    # the Routine "instructions1" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "instructions2" ---
    # create an object to store info about Routine instructions2
    instructions2 = data.Routine(
        name='instructions2',
        components=[instructions2_text, instruction2_key],
    )
    instructions2.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for instruction2_key
    instruction2_key.keys = []
    instruction2_key.rt = []
    _instruction2_key_allKeys = []
    # allowedKeys looks like a variable, so make sure it exists locally
    if 'all_keys' in globals():
        all_keys = globals()['all_keys']
    # store start times for instructions2
    instructions2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instructions2.tStart = globalClock.getTime(format='float')
    instructions2.status = STARTED
    thisExp.addData('instructions2.started', instructions2.tStart)
    instructions2.maxDuration = None
    # keep track of which components have finished
    instructions2Components = instructions2.components
    for thisComponent in instructions2.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instructions2" ---
    instructions2.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *instructions2_text* updates
        
        # if instructions2_text is starting this frame...
        if instructions2_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instructions2_text.frameNStart = frameN  # exact frame index
            instructions2_text.tStart = t  # local t and not account for scr refresh
            instructions2_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instructions2_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instructions2_text.started')
            # update status
            instructions2_text.status = STARTED
            instructions2_text.setAutoDraw(True)
        
        # if instructions2_text is active this frame...
        if instructions2_text.status == STARTED:
            # update params
            pass
        
        # *instruction2_key* updates
        waitOnFlip = False
        
        # if instruction2_key is starting this frame...
        if instruction2_key.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instruction2_key.frameNStart = frameN  # exact frame index
            instruction2_key.tStart = t  # local t and not account for scr refresh
            instruction2_key.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instruction2_key, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instruction2_key.started')
            # update status
            instruction2_key.status = STARTED
            # allowed keys looks like a variable named `all_keys`
            if not type(all_keys) in [list, tuple, np.ndarray]:
                if not isinstance(all_keys, str):
                    all_keys = str(all_keys)
                elif not ',' in all_keys:
                    all_keys = (all_keys,)
                else:
                    all_keys = eval(all_keys)
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(instruction2_key.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(instruction2_key.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if instruction2_key.status == STARTED and not waitOnFlip:
            theseKeys = instruction2_key.getKeys(keyList=list(all_keys), ignoreKeys=["escape"], waitRelease=False)
            _instruction2_key_allKeys.extend(theseKeys)
            if len(_instruction2_key_allKeys):
                instruction2_key.keys = _instruction2_key_allKeys[-1].name  # just the last key pressed
                instruction2_key.rt = _instruction2_key_allKeys[-1].rt
                instruction2_key.duration = _instruction2_key_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            instructions2.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instructions2.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instructions2" ---
    for thisComponent in instructions2.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instructions2
    instructions2.tStop = globalClock.getTime(format='float')
    instructions2.tStopRefresh = tThisFlipGlobal
    thisExp.addData('instructions2.stopped', instructions2.tStop)
    # check responses
    if instruction2_key.keys in ['', [], None]:  # No response was made
        instruction2_key.keys = None
    thisExp.addData('instruction2_key.keys',instruction2_key.keys)
    if instruction2_key.keys != None:  # we had a response
        thisExp.addData('instruction2_key.rt', instruction2_key.rt)
        thisExp.addData('instruction2_key.duration', instruction2_key.duration)
    thisExp.nextEntry()
    # the Routine "instructions2" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "instructions3" ---
    # create an object to store info about Routine instructions3
    instructions3 = data.Routine(
        name='instructions3',
        components=[instructions3_text, instruction3_key],
    )
    instructions3.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for instruction3_key
    instruction3_key.keys = []
    instruction3_key.rt = []
    _instruction3_key_allKeys = []
    # allowedKeys looks like a variable, so make sure it exists locally
    if 'all_keys' in globals():
        all_keys = globals()['all_keys']
    # store start times for instructions3
    instructions3.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instructions3.tStart = globalClock.getTime(format='float')
    instructions3.status = STARTED
    thisExp.addData('instructions3.started', instructions3.tStart)
    instructions3.maxDuration = None
    # keep track of which components have finished
    instructions3Components = instructions3.components
    for thisComponent in instructions3.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instructions3" ---
    instructions3.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *instructions3_text* updates
        
        # if instructions3_text is starting this frame...
        if instructions3_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instructions3_text.frameNStart = frameN  # exact frame index
            instructions3_text.tStart = t  # local t and not account for scr refresh
            instructions3_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instructions3_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instructions3_text.started')
            # update status
            instructions3_text.status = STARTED
            instructions3_text.setAutoDraw(True)
        
        # if instructions3_text is active this frame...
        if instructions3_text.status == STARTED:
            # update params
            pass
        
        # *instruction3_key* updates
        waitOnFlip = False
        
        # if instruction3_key is starting this frame...
        if instruction3_key.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instruction3_key.frameNStart = frameN  # exact frame index
            instruction3_key.tStart = t  # local t and not account for scr refresh
            instruction3_key.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instruction3_key, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instruction3_key.started')
            # update status
            instruction3_key.status = STARTED
            # allowed keys looks like a variable named `all_keys`
            if not type(all_keys) in [list, tuple, np.ndarray]:
                if not isinstance(all_keys, str):
                    all_keys = str(all_keys)
                elif not ',' in all_keys:
                    all_keys = (all_keys,)
                else:
                    all_keys = eval(all_keys)
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(instruction3_key.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(instruction3_key.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if instruction3_key.status == STARTED and not waitOnFlip:
            theseKeys = instruction3_key.getKeys(keyList=list(all_keys), ignoreKeys=["escape"], waitRelease=False)
            _instruction3_key_allKeys.extend(theseKeys)
            if len(_instruction3_key_allKeys):
                instruction3_key.keys = _instruction3_key_allKeys[-1].name  # just the last key pressed
                instruction3_key.rt = _instruction3_key_allKeys[-1].rt
                instruction3_key.duration = _instruction3_key_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            instructions3.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instructions3.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instructions3" ---
    for thisComponent in instructions3.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instructions3
    instructions3.tStop = globalClock.getTime(format='float')
    instructions3.tStopRefresh = tThisFlipGlobal
    thisExp.addData('instructions3.stopped', instructions3.tStop)
    # check responses
    if instruction3_key.keys in ['', [], None]:  # No response was made
        instruction3_key.keys = None
    thisExp.addData('instruction3_key.keys',instruction3_key.keys)
    if instruction3_key.keys != None:  # we had a response
        thisExp.addData('instruction3_key.rt', instruction3_key.rt)
        thisExp.addData('instruction3_key.duration', instruction3_key.duration)
    thisExp.nextEntry()
    # the Routine "instructions3" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "instructions4" ---
    # create an object to store info about Routine instructions4
    instructions4 = data.Routine(
        name='instructions4',
        components=[instructions4_text, instruction4_key],
    )
    instructions4.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for instruction4_key
    instruction4_key.keys = []
    instruction4_key.rt = []
    _instruction4_key_allKeys = []
    # allowedKeys looks like a variable, so make sure it exists locally
    if 'all_keys' in globals():
        all_keys = globals()['all_keys']
    # store start times for instructions4
    instructions4.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instructions4.tStart = globalClock.getTime(format='float')
    instructions4.status = STARTED
    thisExp.addData('instructions4.started', instructions4.tStart)
    instructions4.maxDuration = None
    # keep track of which components have finished
    instructions4Components = instructions4.components
    for thisComponent in instructions4.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instructions4" ---
    instructions4.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *instructions4_text* updates
        
        # if instructions4_text is starting this frame...
        if instructions4_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instructions4_text.frameNStart = frameN  # exact frame index
            instructions4_text.tStart = t  # local t and not account for scr refresh
            instructions4_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instructions4_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instructions4_text.started')
            # update status
            instructions4_text.status = STARTED
            instructions4_text.setAutoDraw(True)
        
        # if instructions4_text is active this frame...
        if instructions4_text.status == STARTED:
            # update params
            pass
        
        # *instruction4_key* updates
        waitOnFlip = False
        
        # if instruction4_key is starting this frame...
        if instruction4_key.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instruction4_key.frameNStart = frameN  # exact frame index
            instruction4_key.tStart = t  # local t and not account for scr refresh
            instruction4_key.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instruction4_key, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instruction4_key.started')
            # update status
            instruction4_key.status = STARTED
            # allowed keys looks like a variable named `all_keys`
            if not type(all_keys) in [list, tuple, np.ndarray]:
                if not isinstance(all_keys, str):
                    all_keys = str(all_keys)
                elif not ',' in all_keys:
                    all_keys = (all_keys,)
                else:
                    all_keys = eval(all_keys)
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(instruction4_key.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(instruction4_key.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if instruction4_key.status == STARTED and not waitOnFlip:
            theseKeys = instruction4_key.getKeys(keyList=list(all_keys), ignoreKeys=["escape"], waitRelease=False)
            _instruction4_key_allKeys.extend(theseKeys)
            if len(_instruction4_key_allKeys):
                instruction4_key.keys = _instruction4_key_allKeys[-1].name  # just the last key pressed
                instruction4_key.rt = _instruction4_key_allKeys[-1].rt
                instruction4_key.duration = _instruction4_key_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            instructions4.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instructions4.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instructions4" ---
    for thisComponent in instructions4.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instructions4
    instructions4.tStop = globalClock.getTime(format='float')
    instructions4.tStopRefresh = tThisFlipGlobal
    thisExp.addData('instructions4.stopped', instructions4.tStop)
    # check responses
    if instruction4_key.keys in ['', [], None]:  # No response was made
        instruction4_key.keys = None
    thisExp.addData('instruction4_key.keys',instruction4_key.keys)
    if instruction4_key.keys != None:  # we had a response
        thisExp.addData('instruction4_key.rt', instruction4_key.rt)
        thisExp.addData('instruction4_key.duration', instruction4_key.duration)
    thisExp.nextEntry()
    # the Routine "instructions4" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "instructions5" ---
    # create an object to store info about Routine instructions5
    instructions5 = data.Routine(
        name='instructions5',
        components=[instructions5_text, instruction5_key],
    )
    instructions5.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for instruction5_key
    instruction5_key.keys = []
    instruction5_key.rt = []
    _instruction5_key_allKeys = []
    # allowedKeys looks like a variable, so make sure it exists locally
    if 'all_keys' in globals():
        all_keys = globals()['all_keys']
    # store start times for instructions5
    instructions5.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instructions5.tStart = globalClock.getTime(format='float')
    instructions5.status = STARTED
    thisExp.addData('instructions5.started', instructions5.tStart)
    instructions5.maxDuration = None
    # keep track of which components have finished
    instructions5Components = instructions5.components
    for thisComponent in instructions5.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instructions5" ---
    instructions5.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *instructions5_text* updates
        
        # if instructions5_text is starting this frame...
        if instructions5_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instructions5_text.frameNStart = frameN  # exact frame index
            instructions5_text.tStart = t  # local t and not account for scr refresh
            instructions5_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instructions5_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instructions5_text.started')
            # update status
            instructions5_text.status = STARTED
            instructions5_text.setAutoDraw(True)
        
        # if instructions5_text is active this frame...
        if instructions5_text.status == STARTED:
            # update params
            pass
        
        # *instruction5_key* updates
        waitOnFlip = False
        
        # if instruction5_key is starting this frame...
        if instruction5_key.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instruction5_key.frameNStart = frameN  # exact frame index
            instruction5_key.tStart = t  # local t and not account for scr refresh
            instruction5_key.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instruction5_key, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instruction5_key.started')
            # update status
            instruction5_key.status = STARTED
            # allowed keys looks like a variable named `all_keys`
            if not type(all_keys) in [list, tuple, np.ndarray]:
                if not isinstance(all_keys, str):
                    all_keys = str(all_keys)
                elif not ',' in all_keys:
                    all_keys = (all_keys,)
                else:
                    all_keys = eval(all_keys)
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(instruction5_key.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(instruction5_key.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if instruction5_key.status == STARTED and not waitOnFlip:
            theseKeys = instruction5_key.getKeys(keyList=list(all_keys), ignoreKeys=["escape"], waitRelease=False)
            _instruction5_key_allKeys.extend(theseKeys)
            if len(_instruction5_key_allKeys):
                instruction5_key.keys = _instruction5_key_allKeys[-1].name  # just the last key pressed
                instruction5_key.rt = _instruction5_key_allKeys[-1].rt
                instruction5_key.duration = _instruction5_key_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            instructions5.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instructions5.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instructions5" ---
    for thisComponent in instructions5.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instructions5
    instructions5.tStop = globalClock.getTime(format='float')
    instructions5.tStopRefresh = tThisFlipGlobal
    thisExp.addData('instructions5.stopped', instructions5.tStop)
    # check responses
    if instruction5_key.keys in ['', [], None]:  # No response was made
        instruction5_key.keys = None
    thisExp.addData('instruction5_key.keys',instruction5_key.keys)
    if instruction5_key.keys != None:  # we had a response
        thisExp.addData('instruction5_key.rt', instruction5_key.rt)
        thisExp.addData('instruction5_key.duration', instruction5_key.duration)
    thisExp.nextEntry()
    # the Routine "instructions5" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    practice_trials = data.TrialHandler2(
        name='practice_trials',
        nReps=1.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('practice.xlsx'), 
        seed=None, 
    )
    thisExp.addLoop(practice_trials)  # add the loop to the experiment
    thisPractice_trial = practice_trials.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisPractice_trial.rgb)
    if thisPractice_trial != None:
        for paramName in thisPractice_trial:
            globals()[paramName] = thisPractice_trial[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisPractice_trial in practice_trials:
        currentLoop = practice_trials
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisPractice_trial.rgb)
        if thisPractice_trial != None:
            for paramName in thisPractice_trial:
                globals()[paramName] = thisPractice_trial[paramName]
        
        # --- Prepare to start Routine "iti_learning_practice" ---
        # create an object to store info about Routine iti_learning_practice
        iti_learning_practice = data.Routine(
            name='iti_learning_practice',
            components=[iti_practice_blank],
        )
        iti_learning_practice.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for iti_learning_practice
        iti_learning_practice.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        iti_learning_practice.tStart = globalClock.getTime(format='float')
        iti_learning_practice.status = STARTED
        thisExp.addData('iti_learning_practice.started', iti_learning_practice.tStart)
        iti_learning_practice.maxDuration = None
        # keep track of which components have finished
        iti_learning_practiceComponents = iti_learning_practice.components
        for thisComponent in iti_learning_practice.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "iti_learning_practice" ---
        # if trial has changed, end Routine now
        if isinstance(practice_trials, data.TrialHandler2) and thisPractice_trial.thisN != practice_trials.thisTrial.thisN:
            continueRoutine = False
        iti_learning_practice.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 2.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *iti_practice_blank* updates
            
            # if iti_practice_blank is starting this frame...
            if iti_practice_blank.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                iti_practice_blank.frameNStart = frameN  # exact frame index
                iti_practice_blank.tStart = t  # local t and not account for scr refresh
                iti_practice_blank.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(iti_practice_blank, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'iti_practice_blank.started')
                # update status
                iti_practice_blank.status = STARTED
                iti_practice_blank.setAutoDraw(True)
            
            # if iti_practice_blank is active this frame...
            if iti_practice_blank.status == STARTED:
                # update params
                pass
            
            # if iti_practice_blank is stopping this frame...
            if iti_practice_blank.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > iti_practice_blank.tStartRefresh + 2.0-frameTolerance:
                    # keep track of stop time/frame for later
                    iti_practice_blank.tStop = t  # not accounting for scr refresh
                    iti_practice_blank.tStopRefresh = tThisFlipGlobal  # on global time
                    iti_practice_blank.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'iti_practice_blank.stopped')
                    # update status
                    iti_practice_blank.status = FINISHED
                    iti_practice_blank.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                iti_learning_practice.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in iti_learning_practice.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "iti_learning_practice" ---
        for thisComponent in iti_learning_practice.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for iti_learning_practice
        iti_learning_practice.tStop = globalClock.getTime(format='float')
        iti_learning_practice.tStopRefresh = tThisFlipGlobal
        thisExp.addData('iti_learning_practice.stopped', iti_learning_practice.tStop)
        # Run 'End Routine' code from set_practice_guess_stim
        guess_cond = practice_list.pop()
        this_cue = cue
        this_target = target
        run_counter = 0
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if iti_learning_practice.maxDurationReached:
            routineTimer.addTime(-iti_learning_practice.maxDuration)
        elif iti_learning_practice.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-2.000000)
        
        # --- Prepare to start Routine "guess_response" ---
        # create an object to store info about Routine guess_response
        guess_response = data.Routine(
            name='guess_response',
            components=[dash_stim_guessing_resp, cue_stim_guessing_resp, guess_stim_resp, guess_question, guess_reached],
        )
        guess_response.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from skip_guess_response
        if guess_cond == "Read":
            trial_dur = 0.000
        else:
            trial_dur = guess_resp_dur
        cue_stim_guessing_resp.setText(this_cue)
        # create starting attributes for guess_reached
        guess_reached.keys = []
        guess_reached.rt = []
        _guess_reached_allKeys = []
        # allowedKeys looks like a variable, so make sure it exists locally
        if 'all_keys' in globals():
            all_keys = globals()['all_keys']
        # store start times for guess_response
        guess_response.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        guess_response.tStart = globalClock.getTime(format='float')
        guess_response.status = STARTED
        thisExp.addData('guess_response.started', guess_response.tStart)
        guess_response.maxDuration = None
        # skip Routine guess_response if its 'Skip if' condition is True
        guess_response.skipped = continueRoutine and not (guess_cond == "Read")
        continueRoutine = guess_response.skipped
        # keep track of which components have finished
        guess_responseComponents = guess_response.components
        for thisComponent in guess_response.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "guess_response" ---
        # if trial has changed, end Routine now
        if isinstance(practice_trials, data.TrialHandler2) and thisPractice_trial.thisN != practice_trials.thisTrial.thisN:
            continueRoutine = False
        guess_response.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *dash_stim_guessing_resp* updates
            
            # if dash_stim_guessing_resp is starting this frame...
            if dash_stim_guessing_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                dash_stim_guessing_resp.frameNStart = frameN  # exact frame index
                dash_stim_guessing_resp.tStart = t  # local t and not account for scr refresh
                dash_stim_guessing_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(dash_stim_guessing_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'dash_stim_guessing_resp.started')
                # update status
                dash_stim_guessing_resp.status = STARTED
                dash_stim_guessing_resp.setAutoDraw(True)
            
            # if dash_stim_guessing_resp is active this frame...
            if dash_stim_guessing_resp.status == STARTED:
                # update params
                pass
            
            # if dash_stim_guessing_resp is stopping this frame...
            if dash_stim_guessing_resp.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > dash_stim_guessing_resp.tStartRefresh + trial_dur-frameTolerance:
                    # keep track of stop time/frame for later
                    dash_stim_guessing_resp.tStop = t  # not accounting for scr refresh
                    dash_stim_guessing_resp.tStopRefresh = tThisFlipGlobal  # on global time
                    dash_stim_guessing_resp.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'dash_stim_guessing_resp.stopped')
                    # update status
                    dash_stim_guessing_resp.status = FINISHED
                    dash_stim_guessing_resp.setAutoDraw(False)
            
            # *cue_stim_guessing_resp* updates
            
            # if cue_stim_guessing_resp is starting this frame...
            if cue_stim_guessing_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                cue_stim_guessing_resp.frameNStart = frameN  # exact frame index
                cue_stim_guessing_resp.tStart = t  # local t and not account for scr refresh
                cue_stim_guessing_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(cue_stim_guessing_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'cue_stim_guessing_resp.started')
                # update status
                cue_stim_guessing_resp.status = STARTED
                cue_stim_guessing_resp.setAutoDraw(True)
            
            # if cue_stim_guessing_resp is active this frame...
            if cue_stim_guessing_resp.status == STARTED:
                # update params
                pass
            
            # if cue_stim_guessing_resp is stopping this frame...
            if cue_stim_guessing_resp.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > cue_stim_guessing_resp.tStartRefresh + trial_dur-frameTolerance:
                    # keep track of stop time/frame for later
                    cue_stim_guessing_resp.tStop = t  # not accounting for scr refresh
                    cue_stim_guessing_resp.tStopRefresh = tThisFlipGlobal  # on global time
                    cue_stim_guessing_resp.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'cue_stim_guessing_resp.stopped')
                    # update status
                    cue_stim_guessing_resp.status = FINISHED
                    cue_stim_guessing_resp.setAutoDraw(False)
            
            # *guess_stim_resp* updates
            
            # if guess_stim_resp is starting this frame...
            if guess_stim_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                guess_stim_resp.frameNStart = frameN  # exact frame index
                guess_stim_resp.tStart = t  # local t and not account for scr refresh
                guess_stim_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(guess_stim_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'guess_stim_resp.started')
                # update status
                guess_stim_resp.status = STARTED
                guess_stim_resp.setAutoDraw(True)
            
            # if guess_stim_resp is active this frame...
            if guess_stim_resp.status == STARTED:
                # update params
                pass
            
            # if guess_stim_resp is stopping this frame...
            if guess_stim_resp.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > guess_stim_resp.tStartRefresh + trial_dur-frameTolerance:
                    # keep track of stop time/frame for later
                    guess_stim_resp.tStop = t  # not accounting for scr refresh
                    guess_stim_resp.tStopRefresh = tThisFlipGlobal  # on global time
                    guess_stim_resp.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'guess_stim_resp.stopped')
                    # update status
                    guess_stim_resp.status = FINISHED
                    guess_stim_resp.setAutoDraw(False)
            
            # *guess_question* updates
            
            # if guess_question is starting this frame...
            if guess_question.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                guess_question.frameNStart = frameN  # exact frame index
                guess_question.tStart = t  # local t and not account for scr refresh
                guess_question.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(guess_question, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'guess_question.started')
                # update status
                guess_question.status = STARTED
                guess_question.setAutoDraw(True)
            
            # if guess_question is active this frame...
            if guess_question.status == STARTED:
                # update params
                pass
            
            # if guess_question is stopping this frame...
            if guess_question.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > guess_question.tStartRefresh + trial_dur-frameTolerance:
                    # keep track of stop time/frame for later
                    guess_question.tStop = t  # not accounting for scr refresh
                    guess_question.tStopRefresh = tThisFlipGlobal  # on global time
                    guess_question.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'guess_question.stopped')
                    # update status
                    guess_question.status = FINISHED
                    guess_question.setAutoDraw(False)
            
            # *guess_reached* updates
            waitOnFlip = False
            
            # if guess_reached is starting this frame...
            if guess_reached.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                guess_reached.frameNStart = frameN  # exact frame index
                guess_reached.tStart = t  # local t and not account for scr refresh
                guess_reached.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(guess_reached, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'guess_reached.started')
                # update status
                guess_reached.status = STARTED
                # allowed keys looks like a variable named `all_keys`
                if not type(all_keys) in [list, tuple, np.ndarray]:
                    if not isinstance(all_keys, str):
                        all_keys = str(all_keys)
                    elif not ',' in all_keys:
                        all_keys = (all_keys,)
                    else:
                        all_keys = eval(all_keys)
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(guess_reached.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(guess_reached.clearEvents, eventType='keyboard')  # clear events on next screen flip
            
            # if guess_reached is stopping this frame...
            if guess_reached.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > guess_reached.tStartRefresh + trial_dur-frameTolerance:
                    # keep track of stop time/frame for later
                    guess_reached.tStop = t  # not accounting for scr refresh
                    guess_reached.tStopRefresh = tThisFlipGlobal  # on global time
                    guess_reached.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'guess_reached.stopped')
                    # update status
                    guess_reached.status = FINISHED
                    guess_reached.status = FINISHED
            if guess_reached.status == STARTED and not waitOnFlip:
                theseKeys = guess_reached.getKeys(keyList=list(all_keys), ignoreKeys=["escape"], waitRelease=False)
                _guess_reached_allKeys.extend(theseKeys)
                if len(_guess_reached_allKeys):
                    guess_reached.keys = _guess_reached_allKeys[-1].name  # just the last key pressed
                    guess_reached.rt = _guess_reached_allKeys[-1].rt
                    guess_reached.duration = _guess_reached_allKeys[-1].duration
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                guess_response.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in guess_response.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "guess_response" ---
        for thisComponent in guess_response.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for guess_response
        guess_response.tStop = globalClock.getTime(format='float')
        guess_response.tStopRefresh = tThisFlipGlobal
        thisExp.addData('guess_response.stopped', guess_response.tStop)
        # check responses
        if guess_reached.keys in ['', [], None]:  # No response was made
            guess_reached.keys = None
        practice_trials.addData('guess_reached.keys',guess_reached.keys)
        if guess_reached.keys != None:  # we had a response
            practice_trials.addData('guess_reached.rt', guess_reached.rt)
            practice_trials.addData('guess_reached.duration', guess_reached.duration)
        # the Routine "guess_response" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "feedback_guess" ---
        # create an object to store info about Routine feedback_guess
        feedback_guess = data.Routine(
            name='feedback_guess',
            components=[no_response_feedback],
        )
        feedback_guess.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from skip_guess_feedback
        if guess_cond == "Read":
            trial_dur = 0.000
        else:
            trial_dur = 1.5
        # Run 'Begin Routine' code from guess_feedback
        feedback = "Ihre Antwort: "
        
        if guess_reached.keys == no_guess_made_key:
            feedback = feedback + "\nKeinen Rateversuch gemacht."
        elif guess_reached.keys == guess_made_key:
            feedback = feedback + "\nRateversuch gemacht."
        else:
            feedback = feedback + "\nKeine Antwort gegeben."
        no_response_feedback.setText(feedback)
        # store start times for feedback_guess
        feedback_guess.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        feedback_guess.tStart = globalClock.getTime(format='float')
        feedback_guess.status = STARTED
        thisExp.addData('feedback_guess.started', feedback_guess.tStart)
        feedback_guess.maxDuration = None
        # skip Routine feedback_guess if its 'Skip if' condition is True
        feedback_guess.skipped = continueRoutine and not (guess_cond == "Read")
        continueRoutine = feedback_guess.skipped
        # keep track of which components have finished
        feedback_guessComponents = feedback_guess.components
        for thisComponent in feedback_guess.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "feedback_guess" ---
        # if trial has changed, end Routine now
        if isinstance(practice_trials, data.TrialHandler2) and thisPractice_trial.thisN != practice_trials.thisTrial.thisN:
            continueRoutine = False
        feedback_guess.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *no_response_feedback* updates
            
            # if no_response_feedback is starting this frame...
            if no_response_feedback.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                no_response_feedback.frameNStart = frameN  # exact frame index
                no_response_feedback.tStart = t  # local t and not account for scr refresh
                no_response_feedback.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(no_response_feedback, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'no_response_feedback.started')
                # update status
                no_response_feedback.status = STARTED
                no_response_feedback.setAutoDraw(True)
            
            # if no_response_feedback is active this frame...
            if no_response_feedback.status == STARTED:
                # update params
                pass
            
            # if no_response_feedback is stopping this frame...
            if no_response_feedback.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > no_response_feedback.tStartRefresh + trial_dur-frameTolerance:
                    # keep track of stop time/frame for later
                    no_response_feedback.tStop = t  # not accounting for scr refresh
                    no_response_feedback.tStopRefresh = tThisFlipGlobal  # on global time
                    no_response_feedback.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'no_response_feedback.stopped')
                    # update status
                    no_response_feedback.status = FINISHED
                    no_response_feedback.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                feedback_guess.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in feedback_guess.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "feedback_guess" ---
        for thisComponent in feedback_guess.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for feedback_guess
        feedback_guess.tStop = globalClock.getTime(format='float')
        feedback_guess.tStopRefresh = tThisFlipGlobal
        thisExp.addData('feedback_guess.stopped', feedback_guess.tStop)
        # the Routine "feedback_guess" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "guess_delay" ---
        # create an object to store info about Routine guess_delay
        guess_delay = data.Routine(
            name='guess_delay',
            components=[guess_delay_cross],
        )
        guess_delay.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from skip_guess_delay
        if guess_cond == "Read":
            trial_dur = 0.000
        else:
            trial_dur = guess_delay_dur
            
        # store start times for guess_delay
        guess_delay.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        guess_delay.tStart = globalClock.getTime(format='float')
        guess_delay.status = STARTED
        thisExp.addData('guess_delay.started', guess_delay.tStart)
        guess_delay.maxDuration = None
        # skip Routine guess_delay if its 'Skip if' condition is True
        guess_delay.skipped = continueRoutine and not (guess_cond == "Read")
        continueRoutine = guess_delay.skipped
        # keep track of which components have finished
        guess_delayComponents = guess_delay.components
        for thisComponent in guess_delay.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "guess_delay" ---
        # if trial has changed, end Routine now
        if isinstance(practice_trials, data.TrialHandler2) and thisPractice_trial.thisN != practice_trials.thisTrial.thisN:
            continueRoutine = False
        guess_delay.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *guess_delay_cross* updates
            
            # if guess_delay_cross is starting this frame...
            if guess_delay_cross.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                guess_delay_cross.frameNStart = frameN  # exact frame index
                guess_delay_cross.tStart = t  # local t and not account for scr refresh
                guess_delay_cross.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(guess_delay_cross, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'guess_delay_cross.started')
                # update status
                guess_delay_cross.status = STARTED
                guess_delay_cross.setAutoDraw(True)
            
            # if guess_delay_cross is active this frame...
            if guess_delay_cross.status == STARTED:
                # update params
                pass
            
            # if guess_delay_cross is stopping this frame...
            if guess_delay_cross.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > guess_delay_cross.tStartRefresh + trial_dur-frameTolerance:
                    # keep track of stop time/frame for later
                    guess_delay_cross.tStop = t  # not accounting for scr refresh
                    guess_delay_cross.tStopRefresh = tThisFlipGlobal  # on global time
                    guess_delay_cross.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'guess_delay_cross.stopped')
                    # update status
                    guess_delay_cross.status = FINISHED
                    guess_delay_cross.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                guess_delay.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in guess_delay.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "guess_delay" ---
        for thisComponent in guess_delay.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for guess_delay
        guess_delay.tStop = globalClock.getTime(format='float')
        guess_delay.tStopRefresh = tThisFlipGlobal
        thisExp.addData('guess_delay.stopped', guess_delay.tStop)
        # the Routine "guess_delay" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "encode" ---
        # create an object to store info about Routine encode
        encode = data.Routine(
            name='encode',
            components=[dash_stim_learning, cue_stim_learning, target_stim, end_encoding],
        )
        encode.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        cue_stim_learning.setText(this_cue)
        target_stim.setText(this_target)
        # create starting attributes for end_encoding
        end_encoding.keys = []
        end_encoding.rt = []
        _end_encoding_allKeys = []
        # Run 'Begin Routine' code from save_learning_trial_type
        cue_types[this_cue] = guess_cond
        target_types[this_target] = guess_cond
        thisExp.addData('trial_type', guess_cond)
        thisExp.addData('Cue', this_cue)
        thisExp.addData('Target', this_target)
        thisExp.addData('RunNr', run_counter)
        # store start times for encode
        encode.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        encode.tStart = globalClock.getTime(format='float')
        encode.status = STARTED
        thisExp.addData('encode.started', encode.tStart)
        encode.maxDuration = None
        # keep track of which components have finished
        encodeComponents = encode.components
        for thisComponent in encode.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "encode" ---
        # if trial has changed, end Routine now
        if isinstance(practice_trials, data.TrialHandler2) and thisPractice_trial.thisN != practice_trials.thisTrial.thisN:
            continueRoutine = False
        encode.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 3.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *dash_stim_learning* updates
            
            # if dash_stim_learning is starting this frame...
            if dash_stim_learning.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                dash_stim_learning.frameNStart = frameN  # exact frame index
                dash_stim_learning.tStart = t  # local t and not account for scr refresh
                dash_stim_learning.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(dash_stim_learning, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'dash_stim_learning.started')
                # update status
                dash_stim_learning.status = STARTED
                dash_stim_learning.setAutoDraw(True)
            
            # if dash_stim_learning is active this frame...
            if dash_stim_learning.status == STARTED:
                # update params
                pass
            
            # if dash_stim_learning is stopping this frame...
            if dash_stim_learning.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > dash_stim_learning.tStartRefresh + 3.0-frameTolerance:
                    # keep track of stop time/frame for later
                    dash_stim_learning.tStop = t  # not accounting for scr refresh
                    dash_stim_learning.tStopRefresh = tThisFlipGlobal  # on global time
                    dash_stim_learning.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'dash_stim_learning.stopped')
                    # update status
                    dash_stim_learning.status = FINISHED
                    dash_stim_learning.setAutoDraw(False)
            
            # *cue_stim_learning* updates
            
            # if cue_stim_learning is starting this frame...
            if cue_stim_learning.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                cue_stim_learning.frameNStart = frameN  # exact frame index
                cue_stim_learning.tStart = t  # local t and not account for scr refresh
                cue_stim_learning.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(cue_stim_learning, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'cue_stim_learning.started')
                # update status
                cue_stim_learning.status = STARTED
                cue_stim_learning.setAutoDraw(True)
            
            # if cue_stim_learning is active this frame...
            if cue_stim_learning.status == STARTED:
                # update params
                pass
            
            # if cue_stim_learning is stopping this frame...
            if cue_stim_learning.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > cue_stim_learning.tStartRefresh + 3.0-frameTolerance:
                    # keep track of stop time/frame for later
                    cue_stim_learning.tStop = t  # not accounting for scr refresh
                    cue_stim_learning.tStopRefresh = tThisFlipGlobal  # on global time
                    cue_stim_learning.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'cue_stim_learning.stopped')
                    # update status
                    cue_stim_learning.status = FINISHED
                    cue_stim_learning.setAutoDraw(False)
            
            # *target_stim* updates
            
            # if target_stim is starting this frame...
            if target_stim.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                target_stim.frameNStart = frameN  # exact frame index
                target_stim.tStart = t  # local t and not account for scr refresh
                target_stim.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(target_stim, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'target_stim.started')
                # update status
                target_stim.status = STARTED
                target_stim.setAutoDraw(True)
            
            # if target_stim is active this frame...
            if target_stim.status == STARTED:
                # update params
                pass
            
            # if target_stim is stopping this frame...
            if target_stim.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > target_stim.tStartRefresh + 3.0-frameTolerance:
                    # keep track of stop time/frame for later
                    target_stim.tStop = t  # not accounting for scr refresh
                    target_stim.tStopRefresh = tThisFlipGlobal  # on global time
                    target_stim.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'target_stim.stopped')
                    # update status
                    target_stim.status = FINISHED
                    target_stim.setAutoDraw(False)
            
            # *end_encoding* updates
            waitOnFlip = False
            
            # if end_encoding is starting this frame...
            if end_encoding.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                end_encoding.frameNStart = frameN  # exact frame index
                end_encoding.tStart = t  # local t and not account for scr refresh
                end_encoding.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(end_encoding, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'end_encoding.started')
                # update status
                end_encoding.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(end_encoding.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(end_encoding.clearEvents, eventType='keyboard')  # clear events on next screen flip
            
            # if end_encoding is stopping this frame...
            if end_encoding.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > end_encoding.tStartRefresh + 3.00-frameTolerance:
                    # keep track of stop time/frame for later
                    end_encoding.tStop = t  # not accounting for scr refresh
                    end_encoding.tStopRefresh = tThisFlipGlobal  # on global time
                    end_encoding.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'end_encoding.stopped')
                    # update status
                    end_encoding.status = FINISHED
                    end_encoding.status = FINISHED
            if end_encoding.status == STARTED and not waitOnFlip:
                theseKeys = end_encoding.getKeys(keyList=['p'], ignoreKeys=["escape"], waitRelease=False)
                _end_encoding_allKeys.extend(theseKeys)
                if len(_end_encoding_allKeys):
                    end_encoding.keys = _end_encoding_allKeys[-1].name  # just the last key pressed
                    end_encoding.rt = _end_encoding_allKeys[-1].rt
                    end_encoding.duration = _end_encoding_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                encode.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in encode.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "encode" ---
        for thisComponent in encode.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for encode
        encode.tStop = globalClock.getTime(format='float')
        encode.tStopRefresh = tThisFlipGlobal
        thisExp.addData('encode.stopped', encode.tStop)
        # check responses
        if end_encoding.keys in ['', [], None]:  # No response was made
            end_encoding.keys = None
        practice_trials.addData('end_encoding.keys',end_encoding.keys)
        if end_encoding.keys != None:  # we had a response
            practice_trials.addData('end_encoding.rt', end_encoding.rt)
            practice_trials.addData('end_encoding.duration', end_encoding.duration)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if encode.maxDurationReached:
            routineTimer.addTime(-encode.maxDuration)
        elif encode.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-3.000000)
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'practice_trials'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # set up handler to look after randomisation of conditions etc
    learning_words_loop = data.TrialHandler2(
        name='learning_words_loop',
        nReps=1.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('cue_target_pairs.xlsx'), 
        seed=None, 
    )
    thisExp.addLoop(learning_words_loop)  # add the loop to the experiment
    thisLearning_words_loop = learning_words_loop.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisLearning_words_loop.rgb)
    if thisLearning_words_loop != None:
        for paramName in thisLearning_words_loop:
            globals()[paramName] = thisLearning_words_loop[paramName]
    
    for thisLearning_words_loop in learning_words_loop:
        currentLoop = learning_words_loop
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        # abbreviate parameter names if possible (e.g. rgb = thisLearning_words_loop.rgb)
        if thisLearning_words_loop != None:
            for paramName in thisLearning_words_loop:
                globals()[paramName] = thisLearning_words_loop[paramName]
        
        # --- Prepare to start Routine "load_word_pairs" ---
        # create an object to store info about Routine load_word_pairs
        load_word_pairs = data.Routine(
            name='load_word_pairs',
            components=[],
        )
        load_word_pairs.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from load_stimuli
        cue_list.append(cue)
        target_list.append(target)
        
        # store start times for load_word_pairs
        load_word_pairs.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        load_word_pairs.tStart = globalClock.getTime(format='float')
        load_word_pairs.status = STARTED
        thisExp.addData('load_word_pairs.started', load_word_pairs.tStart)
        load_word_pairs.maxDuration = None
        # keep track of which components have finished
        load_word_pairsComponents = load_word_pairs.components
        for thisComponent in load_word_pairs.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "load_word_pairs" ---
        # if trial has changed, end Routine now
        if isinstance(learning_words_loop, data.TrialHandler2) and thisLearning_words_loop.thisN != learning_words_loop.thisTrial.thisN:
            continueRoutine = False
        load_word_pairs.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                load_word_pairs.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in load_word_pairs.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "load_word_pairs" ---
        for thisComponent in load_word_pairs.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for load_word_pairs
        load_word_pairs.tStop = globalClock.getTime(format='float')
        load_word_pairs.tStopRefresh = tThisFlipGlobal
        thisExp.addData('load_word_pairs.stopped', load_word_pairs.tStop)
        # the Routine "load_word_pairs" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
    # completed 1.0 repeats of 'learning_words_loop'
    
    
    # --- Prepare to start Routine "start_task" ---
    # create an object to store info about Routine start_task
    start_task = data.Routine(
        name='start_task',
        components=[start_task_text, start_key],
    )
    start_task.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for start_key
    start_key.keys = []
    start_key.rt = []
    _start_key_allKeys = []
    # allowedKeys looks like a variable, so make sure it exists locally
    if 'all_keys' in globals():
        all_keys = globals()['all_keys']
    # Run 'Begin Routine' code from set_up_run_counter
    run_counter = 1
    # store start times for start_task
    start_task.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    start_task.tStart = globalClock.getTime(format='float')
    start_task.status = STARTED
    thisExp.addData('start_task.started', start_task.tStart)
    start_task.maxDuration = None
    # keep track of which components have finished
    start_taskComponents = start_task.components
    for thisComponent in start_task.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "start_task" ---
    start_task.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *start_task_text* updates
        
        # if start_task_text is starting this frame...
        if start_task_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            start_task_text.frameNStart = frameN  # exact frame index
            start_task_text.tStart = t  # local t and not account for scr refresh
            start_task_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(start_task_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'start_task_text.started')
            # update status
            start_task_text.status = STARTED
            start_task_text.setAutoDraw(True)
        
        # if start_task_text is active this frame...
        if start_task_text.status == STARTED:
            # update params
            pass
        
        # *start_key* updates
        waitOnFlip = False
        
        # if start_key is starting this frame...
        if start_key.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            start_key.frameNStart = frameN  # exact frame index
            start_key.tStart = t  # local t and not account for scr refresh
            start_key.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(start_key, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'start_key.started')
            # update status
            start_key.status = STARTED
            # allowed keys looks like a variable named `all_keys`
            if not type(all_keys) in [list, tuple, np.ndarray]:
                if not isinstance(all_keys, str):
                    all_keys = str(all_keys)
                elif not ',' in all_keys:
                    all_keys = (all_keys,)
                else:
                    all_keys = eval(all_keys)
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(start_key.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(start_key.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if start_key.status == STARTED and not waitOnFlip:
            theseKeys = start_key.getKeys(keyList=list(all_keys), ignoreKeys=["escape"], waitRelease=False)
            _start_key_allKeys.extend(theseKeys)
            if len(_start_key_allKeys):
                start_key.keys = _start_key_allKeys[-1].name  # just the last key pressed
                start_key.rt = _start_key_allKeys[-1].rt
                start_key.duration = _start_key_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            start_task.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in start_task.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "start_task" ---
    for thisComponent in start_task.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for start_task
    start_task.tStop = globalClock.getTime(format='float')
    start_task.tStopRefresh = tThisFlipGlobal
    thisExp.addData('start_task.stopped', start_task.tStop)
    # check responses
    if start_key.keys in ['', [], None]:  # No response was made
        start_key.keys = None
    thisExp.addData('start_key.keys',start_key.keys)
    if start_key.keys != None:  # we had a response
        thisExp.addData('start_key.rt', start_key.rt)
        thisExp.addData('start_key.duration', start_key.duration)
    thisExp.nextEntry()
    # the Routine "start_task" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    learning_block = data.TrialHandler2(
        name='learning_block',
        nReps=2.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=[None], 
        seed=None, 
    )
    thisExp.addLoop(learning_block)  # add the loop to the experiment
    thisLearning_block = learning_block.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisLearning_block.rgb)
    if thisLearning_block != None:
        for paramName in thisLearning_block:
            globals()[paramName] = thisLearning_block[paramName]
    
    for thisLearning_block in learning_block:
        currentLoop = learning_block
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        # abbreviate parameter names if possible (e.g. rgb = thisLearning_block.rgb)
        if thisLearning_block != None:
            for paramName in thisLearning_block:
                globals()[paramName] = thisLearning_block[paramName]
        
        # set up handler to look after randomisation of conditions etc
        learning_conditions_loop = data.TrialHandler2(
            name='learning_conditions_loop',
            nReps=1.0, 
            method='random', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=data.importConditions('guess_condition_randomization.xlsx'), 
            seed=None, 
        )
        thisExp.addLoop(learning_conditions_loop)  # add the loop to the experiment
        thisLearning_conditions_loop = learning_conditions_loop.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisLearning_conditions_loop.rgb)
        if thisLearning_conditions_loop != None:
            for paramName in thisLearning_conditions_loop:
                globals()[paramName] = thisLearning_conditions_loop[paramName]
        
        for thisLearning_conditions_loop in learning_conditions_loop:
            currentLoop = learning_conditions_loop
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            # abbreviate parameter names if possible (e.g. rgb = thisLearning_conditions_loop.rgb)
            if thisLearning_conditions_loop != None:
                for paramName in thisLearning_conditions_loop:
                    globals()[paramName] = thisLearning_conditions_loop[paramName]
            
            # --- Prepare to start Routine "load_guess_conditions" ---
            # create an object to store info about Routine load_guess_conditions
            load_guess_conditions = data.Routine(
                name='load_guess_conditions',
                components=[],
            )
            load_guess_conditions.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from load_guess_list
            guess_list.append(guess_condition)
            
            # store start times for load_guess_conditions
            load_guess_conditions.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            load_guess_conditions.tStart = globalClock.getTime(format='float')
            load_guess_conditions.status = STARTED
            thisExp.addData('load_guess_conditions.started', load_guess_conditions.tStart)
            load_guess_conditions.maxDuration = None
            # keep track of which components have finished
            load_guess_conditionsComponents = load_guess_conditions.components
            for thisComponent in load_guess_conditions.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "load_guess_conditions" ---
            # if trial has changed, end Routine now
            if isinstance(learning_conditions_loop, data.TrialHandler2) and thisLearning_conditions_loop.thisN != learning_conditions_loop.thisTrial.thisN:
                continueRoutine = False
            load_guess_conditions.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    load_guess_conditions.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in load_guess_conditions.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "load_guess_conditions" ---
            for thisComponent in load_guess_conditions.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for load_guess_conditions
            load_guess_conditions.tStop = globalClock.getTime(format='float')
            load_guess_conditions.tStopRefresh = tThisFlipGlobal
            thisExp.addData('load_guess_conditions.stopped', load_guess_conditions.tStop)
            # the Routine "load_guess_conditions" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
        # completed 1.0 repeats of 'learning_conditions_loop'
        
        
        # set up handler to look after randomisation of conditions etc
        iti_learning_loop = data.TrialHandler2(
            name='iti_learning_loop',
            nReps=1.0, 
            method='random', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=data.importConditions('iti_randomization.csv'), 
            seed=None, 
        )
        thisExp.addLoop(iti_learning_loop)  # add the loop to the experiment
        thisIti_learning_loop = iti_learning_loop.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisIti_learning_loop.rgb)
        if thisIti_learning_loop != None:
            for paramName in thisIti_learning_loop:
                globals()[paramName] = thisIti_learning_loop[paramName]
        
        for thisIti_learning_loop in iti_learning_loop:
            currentLoop = iti_learning_loop
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            # abbreviate parameter names if possible (e.g. rgb = thisIti_learning_loop.rgb)
            if thisIti_learning_loop != None:
                for paramName in thisIti_learning_loop:
                    globals()[paramName] = thisIti_learning_loop[paramName]
            
            # --- Prepare to start Routine "load_iti" ---
            # create an object to store info about Routine load_iti
            load_iti = data.Routine(
                name='load_iti',
                components=[],
            )
            load_iti.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from load_iti_code
            iti_list.append(iti)
            
            # store start times for load_iti
            load_iti.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            load_iti.tStart = globalClock.getTime(format='float')
            load_iti.status = STARTED
            thisExp.addData('load_iti.started', load_iti.tStart)
            load_iti.maxDuration = None
            # keep track of which components have finished
            load_itiComponents = load_iti.components
            for thisComponent in load_iti.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "load_iti" ---
            # if trial has changed, end Routine now
            if isinstance(iti_learning_loop, data.TrialHandler2) and thisIti_learning_loop.thisN != iti_learning_loop.thisTrial.thisN:
                continueRoutine = False
            load_iti.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    load_iti.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in load_iti.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "load_iti" ---
            for thisComponent in load_iti.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for load_iti
            load_iti.tStop = globalClock.getTime(format='float')
            load_iti.tStopRefresh = tThisFlipGlobal
            thisExp.addData('load_iti.stopped', load_iti.tStop)
            # the Routine "load_iti" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
        # completed 1.0 repeats of 'iti_learning_loop'
        
        
        # set up handler to look after randomisation of conditions etc
        delay_loading_loop = data.TrialHandler2(
            name='delay_loading_loop',
            nReps=1.0, 
            method='random', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=data.importConditions('guess_random_delay.csv'), 
            seed=None, 
        )
        thisExp.addLoop(delay_loading_loop)  # add the loop to the experiment
        thisDelay_loading_loop = delay_loading_loop.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisDelay_loading_loop.rgb)
        if thisDelay_loading_loop != None:
            for paramName in thisDelay_loading_loop:
                globals()[paramName] = thisDelay_loading_loop[paramName]
        
        for thisDelay_loading_loop in delay_loading_loop:
            currentLoop = delay_loading_loop
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            # abbreviate parameter names if possible (e.g. rgb = thisDelay_loading_loop.rgb)
            if thisDelay_loading_loop != None:
                for paramName in thisDelay_loading_loop:
                    globals()[paramName] = thisDelay_loading_loop[paramName]
            
            # --- Prepare to start Routine "load_delay" ---
            # create an object to store info about Routine load_delay
            load_delay = data.Routine(
                name='load_delay',
                components=[],
            )
            load_delay.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from load_delay_code
            delay_list.append(delay)
            
            # store start times for load_delay
            load_delay.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            load_delay.tStart = globalClock.getTime(format='float')
            load_delay.status = STARTED
            thisExp.addData('load_delay.started', load_delay.tStart)
            load_delay.maxDuration = None
            # keep track of which components have finished
            load_delayComponents = load_delay.components
            for thisComponent in load_delay.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "load_delay" ---
            # if trial has changed, end Routine now
            if isinstance(delay_loading_loop, data.TrialHandler2) and thisDelay_loading_loop.thisN != delay_loading_loop.thisTrial.thisN:
                continueRoutine = False
            load_delay.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    load_delay.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in load_delay.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "load_delay" ---
            for thisComponent in load_delay.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for load_delay
            load_delay.tStop = globalClock.getTime(format='float')
            load_delay.tStopRefresh = tThisFlipGlobal
            thisExp.addData('load_delay.stopped', load_delay.tStop)
            # the Routine "load_delay" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
        # completed 1.0 repeats of 'delay_loading_loop'
        
        
        # --- Prepare to start Routine "prep_scanner" ---
        # create an object to store info about Routine prep_scanner
        prep_scanner = data.Routine(
            name='prep_scanner',
            components=[scanner_ready_press, ready_set_text],
        )
        prep_scanner.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # create starting attributes for scanner_ready_press
        scanner_ready_press.keys = []
        scanner_ready_press.rt = []
        _scanner_ready_press_allKeys = []
        # store start times for prep_scanner
        prep_scanner.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        prep_scanner.tStart = globalClock.getTime(format='float')
        prep_scanner.status = STARTED
        thisExp.addData('prep_scanner.started', prep_scanner.tStart)
        prep_scanner.maxDuration = None
        # keep track of which components have finished
        prep_scannerComponents = prep_scanner.components
        for thisComponent in prep_scanner.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "prep_scanner" ---
        # if trial has changed, end Routine now
        if isinstance(learning_block, data.TrialHandler2) and thisLearning_block.thisN != learning_block.thisTrial.thisN:
            continueRoutine = False
        prep_scanner.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *scanner_ready_press* updates
            waitOnFlip = False
            
            # if scanner_ready_press is starting this frame...
            if scanner_ready_press.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                scanner_ready_press.frameNStart = frameN  # exact frame index
                scanner_ready_press.tStart = t  # local t and not account for scr refresh
                scanner_ready_press.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(scanner_ready_press, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'scanner_ready_press.started')
                # update status
                scanner_ready_press.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(scanner_ready_press.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(scanner_ready_press.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if scanner_ready_press.status == STARTED and not waitOnFlip:
                theseKeys = scanner_ready_press.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _scanner_ready_press_allKeys.extend(theseKeys)
                if len(_scanner_ready_press_allKeys):
                    scanner_ready_press.keys = _scanner_ready_press_allKeys[-1].name  # just the last key pressed
                    scanner_ready_press.rt = _scanner_ready_press_allKeys[-1].rt
                    scanner_ready_press.duration = _scanner_ready_press_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # *ready_set_text* updates
            
            # if ready_set_text is starting this frame...
            if ready_set_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                ready_set_text.frameNStart = frameN  # exact frame index
                ready_set_text.tStart = t  # local t and not account for scr refresh
                ready_set_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(ready_set_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'ready_set_text.started')
                # update status
                ready_set_text.status = STARTED
                ready_set_text.setAutoDraw(True)
            
            # if ready_set_text is active this frame...
            if ready_set_text.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                prep_scanner.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in prep_scanner.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "prep_scanner" ---
        for thisComponent in prep_scanner.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for prep_scanner
        prep_scanner.tStop = globalClock.getTime(format='float')
        prep_scanner.tStopRefresh = tThisFlipGlobal
        thisExp.addData('prep_scanner.stopped', prep_scanner.tStop)
        # check responses
        if scanner_ready_press.keys in ['', [], None]:  # No response was made
            scanner_ready_press.keys = None
        learning_block.addData('scanner_ready_press.keys',scanner_ready_press.keys)
        if scanner_ready_press.keys != None:  # we had a response
            learning_block.addData('scanner_ready_press.rt', scanner_ready_press.rt)
            learning_block.addData('scanner_ready_press.duration', scanner_ready_press.duration)
        # the Routine "prep_scanner" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "wait_for_trigger" ---
        # create an object to store info about Routine wait_for_trigger
        wait_for_trigger = data.Routine(
            name='wait_for_trigger',
            components=[starting_soon, skip_trigger],
        )
        wait_for_trigger.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # create starting attributes for skip_trigger
        skip_trigger.keys = []
        skip_trigger.rt = []
        _skip_trigger_allKeys = []
        # store start times for wait_for_trigger
        wait_for_trigger.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        wait_for_trigger.tStart = globalClock.getTime(format='float')
        wait_for_trigger.status = STARTED
        thisExp.addData('wait_for_trigger.started', wait_for_trigger.tStart)
        wait_for_trigger.maxDuration = None
        # keep track of which components have finished
        wait_for_triggerComponents = wait_for_trigger.components
        for thisComponent in wait_for_trigger.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "wait_for_trigger" ---
        # if trial has changed, end Routine now
        if isinstance(learning_block, data.TrialHandler2) and thisLearning_block.thisN != learning_block.thisTrial.thisN:
            continueRoutine = False
        wait_for_trigger.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # Run 'Each Frame' code from catch_trigger
            if expInfo["MRI"] == "1":
                if port.readPin(pinNumber) > 0:
                    continueRoutine = False #A trigger was detected, so move on
            
            # *starting_soon* updates
            
            # if starting_soon is starting this frame...
            if starting_soon.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                starting_soon.frameNStart = frameN  # exact frame index
                starting_soon.tStart = t  # local t and not account for scr refresh
                starting_soon.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(starting_soon, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'starting_soon.started')
                # update status
                starting_soon.status = STARTED
                starting_soon.setAutoDraw(True)
            
            # if starting_soon is active this frame...
            if starting_soon.status == STARTED:
                # update params
                pass
            
            # *skip_trigger* updates
            waitOnFlip = False
            
            # if skip_trigger is starting this frame...
            if skip_trigger.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                skip_trigger.frameNStart = frameN  # exact frame index
                skip_trigger.tStart = t  # local t and not account for scr refresh
                skip_trigger.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(skip_trigger, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'skip_trigger.started')
                # update status
                skip_trigger.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(skip_trigger.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(skip_trigger.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if skip_trigger.status == STARTED and not waitOnFlip:
                theseKeys = skip_trigger.getKeys(keyList=['p'], ignoreKeys=["escape"], waitRelease=False)
                _skip_trigger_allKeys.extend(theseKeys)
                if len(_skip_trigger_allKeys):
                    skip_trigger.keys = _skip_trigger_allKeys[-1].name  # just the last key pressed
                    skip_trigger.rt = _skip_trigger_allKeys[-1].rt
                    skip_trigger.duration = _skip_trigger_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                wait_for_trigger.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in wait_for_trigger.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "wait_for_trigger" ---
        for thisComponent in wait_for_trigger.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for wait_for_trigger
        wait_for_trigger.tStop = globalClock.getTime(format='float')
        wait_for_trigger.tStopRefresh = tThisFlipGlobal
        thisExp.addData('wait_for_trigger.stopped', wait_for_trigger.tStop)
        # check responses
        if skip_trigger.keys in ['', [], None]:  # No response was made
            skip_trigger.keys = None
        learning_block.addData('skip_trigger.keys',skip_trigger.keys)
        if skip_trigger.keys != None:  # we had a response
            learning_block.addData('skip_trigger.rt', skip_trigger.rt)
            learning_block.addData('skip_trigger.duration', skip_trigger.duration)
        # the Routine "wait_for_trigger" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "blank" ---
        # create an object to store info about Routine blank
        blank = data.Routine(
            name='blank',
            components=[begin_end_run_cross, sound_1, sound_2],
        )
        blank.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        sound_1.setSound('bell.wav', secs=1.2, hamming=True)
        sound_1.setVolume(1.0, log=False)
        sound_1.seek(0)
        sound_2.setSound('end_call.wav', secs=1, hamming=True)
        sound_2.setVolume(1.0, log=False)
        sound_2.seek(0)
        # store start times for blank
        blank.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        blank.tStart = globalClock.getTime(format='float')
        blank.status = STARTED
        thisExp.addData('blank.started', blank.tStart)
        blank.maxDuration = None
        # keep track of which components have finished
        blankComponents = blank.components
        for thisComponent in blank.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "blank" ---
        # if trial has changed, end Routine now
        if isinstance(learning_block, data.TrialHandler2) and thisLearning_block.thisN != learning_block.thisTrial.thisN:
            continueRoutine = False
        blank.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 12.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *begin_end_run_cross* updates
            
            # if begin_end_run_cross is starting this frame...
            if begin_end_run_cross.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                begin_end_run_cross.frameNStart = frameN  # exact frame index
                begin_end_run_cross.tStart = t  # local t and not account for scr refresh
                begin_end_run_cross.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(begin_end_run_cross, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'begin_end_run_cross.started')
                # update status
                begin_end_run_cross.status = STARTED
                begin_end_run_cross.setAutoDraw(True)
            
            # if begin_end_run_cross is active this frame...
            if begin_end_run_cross.status == STARTED:
                # update params
                pass
            
            # if begin_end_run_cross is stopping this frame...
            if begin_end_run_cross.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > begin_end_run_cross.tStartRefresh + 12.0-frameTolerance:
                    # keep track of stop time/frame for later
                    begin_end_run_cross.tStop = t  # not accounting for scr refresh
                    begin_end_run_cross.tStopRefresh = tThisFlipGlobal  # on global time
                    begin_end_run_cross.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'begin_end_run_cross.stopped')
                    # update status
                    begin_end_run_cross.status = FINISHED
                    begin_end_run_cross.setAutoDraw(False)
            
            # *sound_1* updates
            
            # if sound_1 is starting this frame...
            if sound_1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                sound_1.frameNStart = frameN  # exact frame index
                sound_1.tStart = t  # local t and not account for scr refresh
                sound_1.tStartRefresh = tThisFlipGlobal  # on global time
                # add timestamp to datafile
                thisExp.addData('sound_1.started', tThisFlipGlobal)
                # update status
                sound_1.status = STARTED
                sound_1.play(when=win)  # sync with win flip
            
            # if sound_1 is stopping this frame...
            if sound_1.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > sound_1.tStartRefresh + 1.2-frameTolerance or sound_1.isFinished:
                    # keep track of stop time/frame for later
                    sound_1.tStop = t  # not accounting for scr refresh
                    sound_1.tStopRefresh = tThisFlipGlobal  # on global time
                    sound_1.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'sound_1.stopped')
                    # update status
                    sound_1.status = FINISHED
                    sound_1.stop()
            
            # *sound_2* updates
            
            # if sound_2 is starting this frame...
            if sound_2.status == NOT_STARTED and tThisFlip >= 10.0-frameTolerance:
                # keep track of start time/frame for later
                sound_2.frameNStart = frameN  # exact frame index
                sound_2.tStart = t  # local t and not account for scr refresh
                sound_2.tStartRefresh = tThisFlipGlobal  # on global time
                # add timestamp to datafile
                thisExp.addData('sound_2.started', tThisFlipGlobal)
                # update status
                sound_2.status = STARTED
                sound_2.play(when=win)  # sync with win flip
            
            # if sound_2 is stopping this frame...
            if sound_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > sound_2.tStartRefresh + 1-frameTolerance or sound_2.isFinished:
                    # keep track of stop time/frame for later
                    sound_2.tStop = t  # not accounting for scr refresh
                    sound_2.tStopRefresh = tThisFlipGlobal  # on global time
                    sound_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'sound_2.stopped')
                    # update status
                    sound_2.status = FINISHED
                    sound_2.stop()
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[sound_1, sound_2]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                blank.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in blank.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "blank" ---
        for thisComponent in blank.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for blank
        blank.tStop = globalClock.getTime(format='float')
        blank.tStopRefresh = tThisFlipGlobal
        thisExp.addData('blank.stopped', blank.tStop)
        sound_1.pause()  # ensure sound has stopped at end of Routine
        sound_2.pause()  # ensure sound has stopped at end of Routine
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if blank.maxDurationReached:
            routineTimer.addTime(-blank.maxDuration)
        elif blank.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-12.000000)
        
        # set up handler to look after randomisation of conditions etc
        learning_trials = data.TrialHandler2(
            name='learning_trials',
            nReps=num_trials, 
            method='random', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=[None], 
            seed=None, 
        )
        thisExp.addLoop(learning_trials)  # add the loop to the experiment
        thisLearning_trial = learning_trials.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisLearning_trial.rgb)
        if thisLearning_trial != None:
            for paramName in thisLearning_trial:
                globals()[paramName] = thisLearning_trial[paramName]
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        for thisLearning_trial in learning_trials:
            currentLoop = learning_trials
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # abbreviate parameter names if possible (e.g. rgb = thisLearning_trial.rgb)
            if thisLearning_trial != None:
                for paramName in thisLearning_trial:
                    globals()[paramName] = thisLearning_trial[paramName]
            
            # --- Prepare to start Routine "setup_learning_trial" ---
            # create an object to store info about Routine setup_learning_trial
            setup_learning_trial = data.Routine(
                name='setup_learning_trial',
                components=[],
            )
            setup_learning_trial.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from set_iti_guess_stim
            guess_cond = guess_list.pop()
            this_iti = iti_list.pop()
            this_cue = cue_list.pop()
            this_target = target_list.pop()
            if guess_cond == "Guess":
                guess_delay_dur = delay_list.pop()
            
            # store start times for setup_learning_trial
            setup_learning_trial.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            setup_learning_trial.tStart = globalClock.getTime(format='float')
            setup_learning_trial.status = STARTED
            thisExp.addData('setup_learning_trial.started', setup_learning_trial.tStart)
            setup_learning_trial.maxDuration = None
            # keep track of which components have finished
            setup_learning_trialComponents = setup_learning_trial.components
            for thisComponent in setup_learning_trial.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "setup_learning_trial" ---
            # if trial has changed, end Routine now
            if isinstance(learning_trials, data.TrialHandler2) and thisLearning_trial.thisN != learning_trials.thisTrial.thisN:
                continueRoutine = False
            setup_learning_trial.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    setup_learning_trial.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in setup_learning_trial.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "setup_learning_trial" ---
            for thisComponent in setup_learning_trial.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for setup_learning_trial
            setup_learning_trial.tStop = globalClock.getTime(format='float')
            setup_learning_trial.tStopRefresh = tThisFlipGlobal
            thisExp.addData('setup_learning_trial.stopped', setup_learning_trial.tStop)
            # the Routine "setup_learning_trial" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "guess_response" ---
            # create an object to store info about Routine guess_response
            guess_response = data.Routine(
                name='guess_response',
                components=[dash_stim_guessing_resp, cue_stim_guessing_resp, guess_stim_resp, guess_question, guess_reached],
            )
            guess_response.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from skip_guess_response
            if guess_cond == "Read":
                trial_dur = 0.000
            else:
                trial_dur = guess_resp_dur
            cue_stim_guessing_resp.setText(this_cue)
            # create starting attributes for guess_reached
            guess_reached.keys = []
            guess_reached.rt = []
            _guess_reached_allKeys = []
            # allowedKeys looks like a variable, so make sure it exists locally
            if 'all_keys' in globals():
                all_keys = globals()['all_keys']
            # store start times for guess_response
            guess_response.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            guess_response.tStart = globalClock.getTime(format='float')
            guess_response.status = STARTED
            thisExp.addData('guess_response.started', guess_response.tStart)
            guess_response.maxDuration = None
            # skip Routine guess_response if its 'Skip if' condition is True
            guess_response.skipped = continueRoutine and not (guess_cond == "Read")
            continueRoutine = guess_response.skipped
            # keep track of which components have finished
            guess_responseComponents = guess_response.components
            for thisComponent in guess_response.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "guess_response" ---
            # if trial has changed, end Routine now
            if isinstance(learning_trials, data.TrialHandler2) and thisLearning_trial.thisN != learning_trials.thisTrial.thisN:
                continueRoutine = False
            guess_response.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *dash_stim_guessing_resp* updates
                
                # if dash_stim_guessing_resp is starting this frame...
                if dash_stim_guessing_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    dash_stim_guessing_resp.frameNStart = frameN  # exact frame index
                    dash_stim_guessing_resp.tStart = t  # local t and not account for scr refresh
                    dash_stim_guessing_resp.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(dash_stim_guessing_resp, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'dash_stim_guessing_resp.started')
                    # update status
                    dash_stim_guessing_resp.status = STARTED
                    dash_stim_guessing_resp.setAutoDraw(True)
                
                # if dash_stim_guessing_resp is active this frame...
                if dash_stim_guessing_resp.status == STARTED:
                    # update params
                    pass
                
                # if dash_stim_guessing_resp is stopping this frame...
                if dash_stim_guessing_resp.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > dash_stim_guessing_resp.tStartRefresh + trial_dur-frameTolerance:
                        # keep track of stop time/frame for later
                        dash_stim_guessing_resp.tStop = t  # not accounting for scr refresh
                        dash_stim_guessing_resp.tStopRefresh = tThisFlipGlobal  # on global time
                        dash_stim_guessing_resp.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'dash_stim_guessing_resp.stopped')
                        # update status
                        dash_stim_guessing_resp.status = FINISHED
                        dash_stim_guessing_resp.setAutoDraw(False)
                
                # *cue_stim_guessing_resp* updates
                
                # if cue_stim_guessing_resp is starting this frame...
                if cue_stim_guessing_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    cue_stim_guessing_resp.frameNStart = frameN  # exact frame index
                    cue_stim_guessing_resp.tStart = t  # local t and not account for scr refresh
                    cue_stim_guessing_resp.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(cue_stim_guessing_resp, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'cue_stim_guessing_resp.started')
                    # update status
                    cue_stim_guessing_resp.status = STARTED
                    cue_stim_guessing_resp.setAutoDraw(True)
                
                # if cue_stim_guessing_resp is active this frame...
                if cue_stim_guessing_resp.status == STARTED:
                    # update params
                    pass
                
                # if cue_stim_guessing_resp is stopping this frame...
                if cue_stim_guessing_resp.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > cue_stim_guessing_resp.tStartRefresh + trial_dur-frameTolerance:
                        # keep track of stop time/frame for later
                        cue_stim_guessing_resp.tStop = t  # not accounting for scr refresh
                        cue_stim_guessing_resp.tStopRefresh = tThisFlipGlobal  # on global time
                        cue_stim_guessing_resp.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'cue_stim_guessing_resp.stopped')
                        # update status
                        cue_stim_guessing_resp.status = FINISHED
                        cue_stim_guessing_resp.setAutoDraw(False)
                
                # *guess_stim_resp* updates
                
                # if guess_stim_resp is starting this frame...
                if guess_stim_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    guess_stim_resp.frameNStart = frameN  # exact frame index
                    guess_stim_resp.tStart = t  # local t and not account for scr refresh
                    guess_stim_resp.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(guess_stim_resp, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'guess_stim_resp.started')
                    # update status
                    guess_stim_resp.status = STARTED
                    guess_stim_resp.setAutoDraw(True)
                
                # if guess_stim_resp is active this frame...
                if guess_stim_resp.status == STARTED:
                    # update params
                    pass
                
                # if guess_stim_resp is stopping this frame...
                if guess_stim_resp.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > guess_stim_resp.tStartRefresh + trial_dur-frameTolerance:
                        # keep track of stop time/frame for later
                        guess_stim_resp.tStop = t  # not accounting for scr refresh
                        guess_stim_resp.tStopRefresh = tThisFlipGlobal  # on global time
                        guess_stim_resp.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'guess_stim_resp.stopped')
                        # update status
                        guess_stim_resp.status = FINISHED
                        guess_stim_resp.setAutoDraw(False)
                
                # *guess_question* updates
                
                # if guess_question is starting this frame...
                if guess_question.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    guess_question.frameNStart = frameN  # exact frame index
                    guess_question.tStart = t  # local t and not account for scr refresh
                    guess_question.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(guess_question, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'guess_question.started')
                    # update status
                    guess_question.status = STARTED
                    guess_question.setAutoDraw(True)
                
                # if guess_question is active this frame...
                if guess_question.status == STARTED:
                    # update params
                    pass
                
                # if guess_question is stopping this frame...
                if guess_question.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > guess_question.tStartRefresh + trial_dur-frameTolerance:
                        # keep track of stop time/frame for later
                        guess_question.tStop = t  # not accounting for scr refresh
                        guess_question.tStopRefresh = tThisFlipGlobal  # on global time
                        guess_question.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'guess_question.stopped')
                        # update status
                        guess_question.status = FINISHED
                        guess_question.setAutoDraw(False)
                
                # *guess_reached* updates
                waitOnFlip = False
                
                # if guess_reached is starting this frame...
                if guess_reached.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    guess_reached.frameNStart = frameN  # exact frame index
                    guess_reached.tStart = t  # local t and not account for scr refresh
                    guess_reached.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(guess_reached, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'guess_reached.started')
                    # update status
                    guess_reached.status = STARTED
                    # allowed keys looks like a variable named `all_keys`
                    if not type(all_keys) in [list, tuple, np.ndarray]:
                        if not isinstance(all_keys, str):
                            all_keys = str(all_keys)
                        elif not ',' in all_keys:
                            all_keys = (all_keys,)
                        else:
                            all_keys = eval(all_keys)
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(guess_reached.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(guess_reached.clearEvents, eventType='keyboard')  # clear events on next screen flip
                
                # if guess_reached is stopping this frame...
                if guess_reached.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > guess_reached.tStartRefresh + trial_dur-frameTolerance:
                        # keep track of stop time/frame for later
                        guess_reached.tStop = t  # not accounting for scr refresh
                        guess_reached.tStopRefresh = tThisFlipGlobal  # on global time
                        guess_reached.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'guess_reached.stopped')
                        # update status
                        guess_reached.status = FINISHED
                        guess_reached.status = FINISHED
                if guess_reached.status == STARTED and not waitOnFlip:
                    theseKeys = guess_reached.getKeys(keyList=list(all_keys), ignoreKeys=["escape"], waitRelease=False)
                    _guess_reached_allKeys.extend(theseKeys)
                    if len(_guess_reached_allKeys):
                        guess_reached.keys = _guess_reached_allKeys[-1].name  # just the last key pressed
                        guess_reached.rt = _guess_reached_allKeys[-1].rt
                        guess_reached.duration = _guess_reached_allKeys[-1].duration
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    guess_response.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in guess_response.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "guess_response" ---
            for thisComponent in guess_response.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for guess_response
            guess_response.tStop = globalClock.getTime(format='float')
            guess_response.tStopRefresh = tThisFlipGlobal
            thisExp.addData('guess_response.stopped', guess_response.tStop)
            # check responses
            if guess_reached.keys in ['', [], None]:  # No response was made
                guess_reached.keys = None
            learning_trials.addData('guess_reached.keys',guess_reached.keys)
            if guess_reached.keys != None:  # we had a response
                learning_trials.addData('guess_reached.rt', guess_reached.rt)
                learning_trials.addData('guess_reached.duration', guess_reached.duration)
            # the Routine "guess_response" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "guess_delay" ---
            # create an object to store info about Routine guess_delay
            guess_delay = data.Routine(
                name='guess_delay',
                components=[guess_delay_cross],
            )
            guess_delay.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from skip_guess_delay
            if guess_cond == "Read":
                trial_dur = 0.000
            else:
                trial_dur = guess_delay_dur
                
            # store start times for guess_delay
            guess_delay.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            guess_delay.tStart = globalClock.getTime(format='float')
            guess_delay.status = STARTED
            thisExp.addData('guess_delay.started', guess_delay.tStart)
            guess_delay.maxDuration = None
            # skip Routine guess_delay if its 'Skip if' condition is True
            guess_delay.skipped = continueRoutine and not (guess_cond == "Read")
            continueRoutine = guess_delay.skipped
            # keep track of which components have finished
            guess_delayComponents = guess_delay.components
            for thisComponent in guess_delay.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "guess_delay" ---
            # if trial has changed, end Routine now
            if isinstance(learning_trials, data.TrialHandler2) and thisLearning_trial.thisN != learning_trials.thisTrial.thisN:
                continueRoutine = False
            guess_delay.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *guess_delay_cross* updates
                
                # if guess_delay_cross is starting this frame...
                if guess_delay_cross.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    guess_delay_cross.frameNStart = frameN  # exact frame index
                    guess_delay_cross.tStart = t  # local t and not account for scr refresh
                    guess_delay_cross.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(guess_delay_cross, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'guess_delay_cross.started')
                    # update status
                    guess_delay_cross.status = STARTED
                    guess_delay_cross.setAutoDraw(True)
                
                # if guess_delay_cross is active this frame...
                if guess_delay_cross.status == STARTED:
                    # update params
                    pass
                
                # if guess_delay_cross is stopping this frame...
                if guess_delay_cross.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > guess_delay_cross.tStartRefresh + trial_dur-frameTolerance:
                        # keep track of stop time/frame for later
                        guess_delay_cross.tStop = t  # not accounting for scr refresh
                        guess_delay_cross.tStopRefresh = tThisFlipGlobal  # on global time
                        guess_delay_cross.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'guess_delay_cross.stopped')
                        # update status
                        guess_delay_cross.status = FINISHED
                        guess_delay_cross.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    guess_delay.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in guess_delay.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "guess_delay" ---
            for thisComponent in guess_delay.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for guess_delay
            guess_delay.tStop = globalClock.getTime(format='float')
            guess_delay.tStopRefresh = tThisFlipGlobal
            thisExp.addData('guess_delay.stopped', guess_delay.tStop)
            # the Routine "guess_delay" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "encode" ---
            # create an object to store info about Routine encode
            encode = data.Routine(
                name='encode',
                components=[dash_stim_learning, cue_stim_learning, target_stim, end_encoding],
            )
            encode.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            cue_stim_learning.setText(this_cue)
            target_stim.setText(this_target)
            # create starting attributes for end_encoding
            end_encoding.keys = []
            end_encoding.rt = []
            _end_encoding_allKeys = []
            # Run 'Begin Routine' code from save_learning_trial_type
            cue_types[this_cue] = guess_cond
            target_types[this_target] = guess_cond
            thisExp.addData('trial_type', guess_cond)
            thisExp.addData('Cue', this_cue)
            thisExp.addData('Target', this_target)
            thisExp.addData('RunNr', run_counter)
            # store start times for encode
            encode.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            encode.tStart = globalClock.getTime(format='float')
            encode.status = STARTED
            thisExp.addData('encode.started', encode.tStart)
            encode.maxDuration = None
            # keep track of which components have finished
            encodeComponents = encode.components
            for thisComponent in encode.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "encode" ---
            # if trial has changed, end Routine now
            if isinstance(learning_trials, data.TrialHandler2) and thisLearning_trial.thisN != learning_trials.thisTrial.thisN:
                continueRoutine = False
            encode.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 3.0:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *dash_stim_learning* updates
                
                # if dash_stim_learning is starting this frame...
                if dash_stim_learning.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    dash_stim_learning.frameNStart = frameN  # exact frame index
                    dash_stim_learning.tStart = t  # local t and not account for scr refresh
                    dash_stim_learning.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(dash_stim_learning, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'dash_stim_learning.started')
                    # update status
                    dash_stim_learning.status = STARTED
                    dash_stim_learning.setAutoDraw(True)
                
                # if dash_stim_learning is active this frame...
                if dash_stim_learning.status == STARTED:
                    # update params
                    pass
                
                # if dash_stim_learning is stopping this frame...
                if dash_stim_learning.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > dash_stim_learning.tStartRefresh + 3.0-frameTolerance:
                        # keep track of stop time/frame for later
                        dash_stim_learning.tStop = t  # not accounting for scr refresh
                        dash_stim_learning.tStopRefresh = tThisFlipGlobal  # on global time
                        dash_stim_learning.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'dash_stim_learning.stopped')
                        # update status
                        dash_stim_learning.status = FINISHED
                        dash_stim_learning.setAutoDraw(False)
                
                # *cue_stim_learning* updates
                
                # if cue_stim_learning is starting this frame...
                if cue_stim_learning.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    cue_stim_learning.frameNStart = frameN  # exact frame index
                    cue_stim_learning.tStart = t  # local t and not account for scr refresh
                    cue_stim_learning.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(cue_stim_learning, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'cue_stim_learning.started')
                    # update status
                    cue_stim_learning.status = STARTED
                    cue_stim_learning.setAutoDraw(True)
                
                # if cue_stim_learning is active this frame...
                if cue_stim_learning.status == STARTED:
                    # update params
                    pass
                
                # if cue_stim_learning is stopping this frame...
                if cue_stim_learning.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > cue_stim_learning.tStartRefresh + 3.0-frameTolerance:
                        # keep track of stop time/frame for later
                        cue_stim_learning.tStop = t  # not accounting for scr refresh
                        cue_stim_learning.tStopRefresh = tThisFlipGlobal  # on global time
                        cue_stim_learning.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'cue_stim_learning.stopped')
                        # update status
                        cue_stim_learning.status = FINISHED
                        cue_stim_learning.setAutoDraw(False)
                
                # *target_stim* updates
                
                # if target_stim is starting this frame...
                if target_stim.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    target_stim.frameNStart = frameN  # exact frame index
                    target_stim.tStart = t  # local t and not account for scr refresh
                    target_stim.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(target_stim, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'target_stim.started')
                    # update status
                    target_stim.status = STARTED
                    target_stim.setAutoDraw(True)
                
                # if target_stim is active this frame...
                if target_stim.status == STARTED:
                    # update params
                    pass
                
                # if target_stim is stopping this frame...
                if target_stim.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > target_stim.tStartRefresh + 3.0-frameTolerance:
                        # keep track of stop time/frame for later
                        target_stim.tStop = t  # not accounting for scr refresh
                        target_stim.tStopRefresh = tThisFlipGlobal  # on global time
                        target_stim.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'target_stim.stopped')
                        # update status
                        target_stim.status = FINISHED
                        target_stim.setAutoDraw(False)
                
                # *end_encoding* updates
                waitOnFlip = False
                
                # if end_encoding is starting this frame...
                if end_encoding.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    end_encoding.frameNStart = frameN  # exact frame index
                    end_encoding.tStart = t  # local t and not account for scr refresh
                    end_encoding.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(end_encoding, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'end_encoding.started')
                    # update status
                    end_encoding.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(end_encoding.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(end_encoding.clearEvents, eventType='keyboard')  # clear events on next screen flip
                
                # if end_encoding is stopping this frame...
                if end_encoding.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > end_encoding.tStartRefresh + 3.00-frameTolerance:
                        # keep track of stop time/frame for later
                        end_encoding.tStop = t  # not accounting for scr refresh
                        end_encoding.tStopRefresh = tThisFlipGlobal  # on global time
                        end_encoding.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'end_encoding.stopped')
                        # update status
                        end_encoding.status = FINISHED
                        end_encoding.status = FINISHED
                if end_encoding.status == STARTED and not waitOnFlip:
                    theseKeys = end_encoding.getKeys(keyList=['p'], ignoreKeys=["escape"], waitRelease=False)
                    _end_encoding_allKeys.extend(theseKeys)
                    if len(_end_encoding_allKeys):
                        end_encoding.keys = _end_encoding_allKeys[-1].name  # just the last key pressed
                        end_encoding.rt = _end_encoding_allKeys[-1].rt
                        end_encoding.duration = _end_encoding_allKeys[-1].duration
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    encode.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in encode.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "encode" ---
            for thisComponent in encode.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for encode
            encode.tStop = globalClock.getTime(format='float')
            encode.tStopRefresh = tThisFlipGlobal
            thisExp.addData('encode.stopped', encode.tStop)
            # check responses
            if end_encoding.keys in ['', [], None]:  # No response was made
                end_encoding.keys = None
            learning_trials.addData('end_encoding.keys',end_encoding.keys)
            if end_encoding.keys != None:  # we had a response
                learning_trials.addData('end_encoding.rt', end_encoding.rt)
                learning_trials.addData('end_encoding.duration', end_encoding.duration)
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if encode.maxDurationReached:
                routineTimer.addTime(-encode.maxDuration)
            elif encode.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-3.000000)
            
            # --- Prepare to start Routine "iti_task" ---
            # create an object to store info about Routine iti_task
            iti_task = data.Routine(
                name='iti_task',
                components=[iti_cross],
            )
            iti_task.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # store start times for iti_task
            iti_task.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            iti_task.tStart = globalClock.getTime(format='float')
            iti_task.status = STARTED
            thisExp.addData('iti_task.started', iti_task.tStart)
            iti_task.maxDuration = None
            # keep track of which components have finished
            iti_taskComponents = iti_task.components
            for thisComponent in iti_task.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "iti_task" ---
            # if trial has changed, end Routine now
            if isinstance(learning_trials, data.TrialHandler2) and thisLearning_trial.thisN != learning_trials.thisTrial.thisN:
                continueRoutine = False
            iti_task.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *iti_cross* updates
                
                # if iti_cross is starting this frame...
                if iti_cross.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    iti_cross.frameNStart = frameN  # exact frame index
                    iti_cross.tStart = t  # local t and not account for scr refresh
                    iti_cross.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(iti_cross, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'iti_cross.started')
                    # update status
                    iti_cross.status = STARTED
                    iti_cross.setAutoDraw(True)
                
                # if iti_cross is active this frame...
                if iti_cross.status == STARTED:
                    # update params
                    pass
                
                # if iti_cross is stopping this frame...
                if iti_cross.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > iti_cross.tStartRefresh + this_iti-frameTolerance:
                        # keep track of stop time/frame for later
                        iti_cross.tStop = t  # not accounting for scr refresh
                        iti_cross.tStopRefresh = tThisFlipGlobal  # on global time
                        iti_cross.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'iti_cross.stopped')
                        # update status
                        iti_cross.status = FINISHED
                        iti_cross.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    iti_task.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in iti_task.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "iti_task" ---
            for thisComponent in iti_task.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for iti_task
            iti_task.tStop = globalClock.getTime(format='float')
            iti_task.tStopRefresh = tThisFlipGlobal
            thisExp.addData('iti_task.stopped', iti_task.tStop)
            # the Routine "iti_task" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            thisExp.nextEntry()
            
        # completed num_trials repeats of 'learning_trials'
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        # --- Prepare to start Routine "blank" ---
        # create an object to store info about Routine blank
        blank = data.Routine(
            name='blank',
            components=[begin_end_run_cross, sound_1, sound_2],
        )
        blank.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        sound_1.setSound('bell.wav', secs=1.2, hamming=True)
        sound_1.setVolume(1.0, log=False)
        sound_1.seek(0)
        sound_2.setSound('end_call.wav', secs=1, hamming=True)
        sound_2.setVolume(1.0, log=False)
        sound_2.seek(0)
        # store start times for blank
        blank.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        blank.tStart = globalClock.getTime(format='float')
        blank.status = STARTED
        thisExp.addData('blank.started', blank.tStart)
        blank.maxDuration = None
        # keep track of which components have finished
        blankComponents = blank.components
        for thisComponent in blank.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "blank" ---
        # if trial has changed, end Routine now
        if isinstance(learning_block, data.TrialHandler2) and thisLearning_block.thisN != learning_block.thisTrial.thisN:
            continueRoutine = False
        blank.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 12.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *begin_end_run_cross* updates
            
            # if begin_end_run_cross is starting this frame...
            if begin_end_run_cross.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                begin_end_run_cross.frameNStart = frameN  # exact frame index
                begin_end_run_cross.tStart = t  # local t and not account for scr refresh
                begin_end_run_cross.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(begin_end_run_cross, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'begin_end_run_cross.started')
                # update status
                begin_end_run_cross.status = STARTED
                begin_end_run_cross.setAutoDraw(True)
            
            # if begin_end_run_cross is active this frame...
            if begin_end_run_cross.status == STARTED:
                # update params
                pass
            
            # if begin_end_run_cross is stopping this frame...
            if begin_end_run_cross.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > begin_end_run_cross.tStartRefresh + 12.0-frameTolerance:
                    # keep track of stop time/frame for later
                    begin_end_run_cross.tStop = t  # not accounting for scr refresh
                    begin_end_run_cross.tStopRefresh = tThisFlipGlobal  # on global time
                    begin_end_run_cross.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'begin_end_run_cross.stopped')
                    # update status
                    begin_end_run_cross.status = FINISHED
                    begin_end_run_cross.setAutoDraw(False)
            
            # *sound_1* updates
            
            # if sound_1 is starting this frame...
            if sound_1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                sound_1.frameNStart = frameN  # exact frame index
                sound_1.tStart = t  # local t and not account for scr refresh
                sound_1.tStartRefresh = tThisFlipGlobal  # on global time
                # add timestamp to datafile
                thisExp.addData('sound_1.started', tThisFlipGlobal)
                # update status
                sound_1.status = STARTED
                sound_1.play(when=win)  # sync with win flip
            
            # if sound_1 is stopping this frame...
            if sound_1.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > sound_1.tStartRefresh + 1.2-frameTolerance or sound_1.isFinished:
                    # keep track of stop time/frame for later
                    sound_1.tStop = t  # not accounting for scr refresh
                    sound_1.tStopRefresh = tThisFlipGlobal  # on global time
                    sound_1.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'sound_1.stopped')
                    # update status
                    sound_1.status = FINISHED
                    sound_1.stop()
            
            # *sound_2* updates
            
            # if sound_2 is starting this frame...
            if sound_2.status == NOT_STARTED and tThisFlip >= 10.0-frameTolerance:
                # keep track of start time/frame for later
                sound_2.frameNStart = frameN  # exact frame index
                sound_2.tStart = t  # local t and not account for scr refresh
                sound_2.tStartRefresh = tThisFlipGlobal  # on global time
                # add timestamp to datafile
                thisExp.addData('sound_2.started', tThisFlipGlobal)
                # update status
                sound_2.status = STARTED
                sound_2.play(when=win)  # sync with win flip
            
            # if sound_2 is stopping this frame...
            if sound_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > sound_2.tStartRefresh + 1-frameTolerance or sound_2.isFinished:
                    # keep track of stop time/frame for later
                    sound_2.tStop = t  # not accounting for scr refresh
                    sound_2.tStopRefresh = tThisFlipGlobal  # on global time
                    sound_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'sound_2.stopped')
                    # update status
                    sound_2.status = FINISHED
                    sound_2.stop()
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[sound_1, sound_2]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                blank.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in blank.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "blank" ---
        for thisComponent in blank.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for blank
        blank.tStop = globalClock.getTime(format='float')
        blank.tStopRefresh = tThisFlipGlobal
        thisExp.addData('blank.stopped', blank.tStop)
        sound_1.pause()  # ensure sound has stopped at end of Routine
        sound_2.pause()  # ensure sound has stopped at end of Routine
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if blank.maxDurationReached:
            routineTimer.addTime(-blank.maxDuration)
        elif blank.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-12.000000)
        
        # --- Prepare to start Routine "task_break" ---
        # create an object to store info about Routine task_break
        task_break = data.Routine(
            name='task_break',
            components=[task_break_text, task_break_resp],
        )
        task_break.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # create starting attributes for task_break_resp
        task_break_resp.keys = []
        task_break_resp.rt = []
        _task_break_resp_allKeys = []
        # allowedKeys looks like a variable, so make sure it exists locally
        if 'all_keys' in globals():
            all_keys = globals()['all_keys']
        # store start times for task_break
        task_break.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        task_break.tStart = globalClock.getTime(format='float')
        task_break.status = STARTED
        thisExp.addData('task_break.started', task_break.tStart)
        task_break.maxDuration = None
        # skip Routine task_break if its 'Skip if' condition is True
        task_break.skipped = continueRoutine and not (run_counter >= 2)
        continueRoutine = task_break.skipped
        # keep track of which components have finished
        task_breakComponents = task_break.components
        for thisComponent in task_break.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "task_break" ---
        # if trial has changed, end Routine now
        if isinstance(learning_block, data.TrialHandler2) and thisLearning_block.thisN != learning_block.thisTrial.thisN:
            continueRoutine = False
        task_break.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *task_break_text* updates
            
            # if task_break_text is starting this frame...
            if task_break_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                task_break_text.frameNStart = frameN  # exact frame index
                task_break_text.tStart = t  # local t and not account for scr refresh
                task_break_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(task_break_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'task_break_text.started')
                # update status
                task_break_text.status = STARTED
                task_break_text.setAutoDraw(True)
            
            # if task_break_text is active this frame...
            if task_break_text.status == STARTED:
                # update params
                pass
            
            # *task_break_resp* updates
            waitOnFlip = False
            
            # if task_break_resp is starting this frame...
            if task_break_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                task_break_resp.frameNStart = frameN  # exact frame index
                task_break_resp.tStart = t  # local t and not account for scr refresh
                task_break_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(task_break_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'task_break_resp.started')
                # update status
                task_break_resp.status = STARTED
                # allowed keys looks like a variable named `all_keys`
                if not type(all_keys) in [list, tuple, np.ndarray]:
                    if not isinstance(all_keys, str):
                        all_keys = str(all_keys)
                    elif not ',' in all_keys:
                        all_keys = (all_keys,)
                    else:
                        all_keys = eval(all_keys)
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(task_break_resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(task_break_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if task_break_resp.status == STARTED and not waitOnFlip:
                theseKeys = task_break_resp.getKeys(keyList=list(all_keys), ignoreKeys=["escape"], waitRelease=False)
                _task_break_resp_allKeys.extend(theseKeys)
                if len(_task_break_resp_allKeys):
                    task_break_resp.keys = _task_break_resp_allKeys[-1].name  # just the last key pressed
                    task_break_resp.rt = _task_break_resp_allKeys[-1].rt
                    task_break_resp.duration = _task_break_resp_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                task_break.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in task_break.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "task_break" ---
        for thisComponent in task_break.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for task_break
        task_break.tStop = globalClock.getTime(format='float')
        task_break.tStopRefresh = tThisFlipGlobal
        thisExp.addData('task_break.stopped', task_break.tStop)
        # check responses
        if task_break_resp.keys in ['', [], None]:  # No response was made
            task_break_resp.keys = None
        learning_block.addData('task_break_resp.keys',task_break_resp.keys)
        if task_break_resp.keys != None:  # we had a response
            learning_block.addData('task_break_resp.rt', task_break_resp.rt)
            learning_block.addData('task_break_resp.duration', task_break_resp.duration)
        # Run 'End Routine' code from task_reset_counter
        run_counter = run_counter + 1
        # the Routine "task_break" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
    # completed 2.0 repeats of 'learning_block'
    
    
    # --- Prepare to start Routine "end_part1" ---
    # create an object to store info about Routine end_part1
    end_part1 = data.Routine(
        name='end_part1',
        components=[end_part1_key, end_part1_text],
    )
    end_part1.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for end_part1_key
    end_part1_key.keys = []
    end_part1_key.rt = []
    _end_part1_key_allKeys = []
    # allowedKeys looks like a variable, so make sure it exists locally
    if 'all_keys' in globals():
        all_keys = globals()['all_keys']
    # store start times for end_part1
    end_part1.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    end_part1.tStart = globalClock.getTime(format='float')
    end_part1.status = STARTED
    thisExp.addData('end_part1.started', end_part1.tStart)
    end_part1.maxDuration = None
    # keep track of which components have finished
    end_part1Components = end_part1.components
    for thisComponent in end_part1.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "end_part1" ---
    end_part1.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *end_part1_key* updates
        waitOnFlip = False
        
        # if end_part1_key is starting this frame...
        if end_part1_key.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            end_part1_key.frameNStart = frameN  # exact frame index
            end_part1_key.tStart = t  # local t and not account for scr refresh
            end_part1_key.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(end_part1_key, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'end_part1_key.started')
            # update status
            end_part1_key.status = STARTED
            # allowed keys looks like a variable named `all_keys`
            if not type(all_keys) in [list, tuple, np.ndarray]:
                if not isinstance(all_keys, str):
                    all_keys = str(all_keys)
                elif not ',' in all_keys:
                    all_keys = (all_keys,)
                else:
                    all_keys = eval(all_keys)
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(end_part1_key.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(end_part1_key.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if end_part1_key.status == STARTED and not waitOnFlip:
            theseKeys = end_part1_key.getKeys(keyList=list(all_keys), ignoreKeys=["escape"], waitRelease=False)
            _end_part1_key_allKeys.extend(theseKeys)
            if len(_end_part1_key_allKeys):
                end_part1_key.keys = _end_part1_key_allKeys[-1].name  # just the last key pressed
                end_part1_key.rt = _end_part1_key_allKeys[-1].rt
                end_part1_key.duration = _end_part1_key_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *end_part1_text* updates
        
        # if end_part1_text is starting this frame...
        if end_part1_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            end_part1_text.frameNStart = frameN  # exact frame index
            end_part1_text.tStart = t  # local t and not account for scr refresh
            end_part1_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(end_part1_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'end_part1_text.started')
            # update status
            end_part1_text.status = STARTED
            end_part1_text.setAutoDraw(True)
        
        # if end_part1_text is active this frame...
        if end_part1_text.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            end_part1.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in end_part1.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "end_part1" ---
    for thisComponent in end_part1.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for end_part1
    end_part1.tStop = globalClock.getTime(format='float')
    end_part1.tStopRefresh = tThisFlipGlobal
    thisExp.addData('end_part1.stopped', end_part1.tStop)
    # check responses
    if end_part1_key.keys in ['', [], None]:  # No response was made
        end_part1_key.keys = None
    thisExp.addData('end_part1_key.keys',end_part1_key.keys)
    if end_part1_key.keys != None:  # we had a response
        thisExp.addData('end_part1_key.rt', end_part1_key.rt)
        thisExp.addData('end_part1_key.duration', end_part1_key.duration)
    thisExp.nextEntry()
    # the Routine "end_part1" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "distractor_instruction" ---
    # create an object to store info about Routine distractor_instruction
    distractor_instruction = data.Routine(
        name='distractor_instruction',
        components=[dist_instruc, key_resp_dist_instr],
    )
    distractor_instruction.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_dist_instr
    key_resp_dist_instr.keys = []
    key_resp_dist_instr.rt = []
    _key_resp_dist_instr_allKeys = []
    # store start times for distractor_instruction
    distractor_instruction.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    distractor_instruction.tStart = globalClock.getTime(format='float')
    distractor_instruction.status = STARTED
    thisExp.addData('distractor_instruction.started', distractor_instruction.tStart)
    distractor_instruction.maxDuration = None
    # keep track of which components have finished
    distractor_instructionComponents = distractor_instruction.components
    for thisComponent in distractor_instruction.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "distractor_instruction" ---
    distractor_instruction.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *dist_instruc* updates
        
        # if dist_instruc is starting this frame...
        if dist_instruc.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            dist_instruc.frameNStart = frameN  # exact frame index
            dist_instruc.tStart = t  # local t and not account for scr refresh
            dist_instruc.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(dist_instruc, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'dist_instruc.started')
            # update status
            dist_instruc.status = STARTED
            dist_instruc.setAutoDraw(True)
        
        # if dist_instruc is active this frame...
        if dist_instruc.status == STARTED:
            # update params
            pass
        
        # *key_resp_dist_instr* updates
        waitOnFlip = False
        
        # if key_resp_dist_instr is starting this frame...
        if key_resp_dist_instr.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
            # keep track of start time/frame for later
            key_resp_dist_instr.frameNStart = frameN  # exact frame index
            key_resp_dist_instr.tStart = t  # local t and not account for scr refresh
            key_resp_dist_instr.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_dist_instr, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_dist_instr.started')
            # update status
            key_resp_dist_instr.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_dist_instr.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_dist_instr.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_dist_instr.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_dist_instr.getKeys(keyList=['2', '3', '4', '5'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_dist_instr_allKeys.extend(theseKeys)
            if len(_key_resp_dist_instr_allKeys):
                key_resp_dist_instr.keys = _key_resp_dist_instr_allKeys[-1].name  # just the last key pressed
                key_resp_dist_instr.rt = _key_resp_dist_instr_allKeys[-1].rt
                key_resp_dist_instr.duration = _key_resp_dist_instr_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            distractor_instruction.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in distractor_instruction.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "distractor_instruction" ---
    for thisComponent in distractor_instruction.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for distractor_instruction
    distractor_instruction.tStop = globalClock.getTime(format='float')
    distractor_instruction.tStopRefresh = tThisFlipGlobal
    thisExp.addData('distractor_instruction.stopped', distractor_instruction.tStop)
    # check responses
    if key_resp_dist_instr.keys in ['', [], None]:  # No response was made
        key_resp_dist_instr.keys = None
    thisExp.addData('key_resp_dist_instr.keys',key_resp_dist_instr.keys)
    if key_resp_dist_instr.keys != None:  # we had a response
        thisExp.addData('key_resp_dist_instr.rt', key_resp_dist_instr.rt)
        thisExp.addData('key_resp_dist_instr.duration', key_resp_dist_instr.duration)
    # Run 'End Routine' code from dist_clock1
    startTime = core.getTime()
    thisExp.nextEntry()
    # the Routine "distractor_instruction" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "start_dist_prac" ---
    # create an object to store info about Routine start_dist_prac
    start_dist_prac = data.Routine(
        name='start_dist_prac',
        components=[start_dist_prac_text],
    )
    start_dist_prac.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # Run 'Begin Routine' code from dist_pr_clock
    startTime = core.getTime()
    dt = 0
    # store start times for start_dist_prac
    start_dist_prac.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    start_dist_prac.tStart = globalClock.getTime(format='float')
    start_dist_prac.status = STARTED
    thisExp.addData('start_dist_prac.started', start_dist_prac.tStart)
    start_dist_prac.maxDuration = None
    # keep track of which components have finished
    start_dist_pracComponents = start_dist_prac.components
    for thisComponent in start_dist_prac.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "start_dist_prac" ---
    start_dist_prac.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 2.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *start_dist_prac_text* updates
        
        # if start_dist_prac_text is starting this frame...
        if start_dist_prac_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            start_dist_prac_text.frameNStart = frameN  # exact frame index
            start_dist_prac_text.tStart = t  # local t and not account for scr refresh
            start_dist_prac_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(start_dist_prac_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'start_dist_prac_text.started')
            # update status
            start_dist_prac_text.status = STARTED
            start_dist_prac_text.setAutoDraw(True)
        
        # if start_dist_prac_text is active this frame...
        if start_dist_prac_text.status == STARTED:
            # update params
            pass
        
        # if start_dist_prac_text is stopping this frame...
        if start_dist_prac_text.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > start_dist_prac_text.tStartRefresh + 2.0-frameTolerance:
                # keep track of stop time/frame for later
                start_dist_prac_text.tStop = t  # not accounting for scr refresh
                start_dist_prac_text.tStopRefresh = tThisFlipGlobal  # on global time
                start_dist_prac_text.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'start_dist_prac_text.stopped')
                # update status
                start_dist_prac_text.status = FINISHED
                start_dist_prac_text.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            start_dist_prac.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in start_dist_prac.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "start_dist_prac" ---
    for thisComponent in start_dist_prac.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for start_dist_prac
    start_dist_prac.tStop = globalClock.getTime(format='float')
    start_dist_prac.tStopRefresh = tThisFlipGlobal
    thisExp.addData('start_dist_prac.stopped', start_dist_prac.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if start_dist_prac.maxDurationReached:
        routineTimer.addTime(-start_dist_prac.maxDuration)
    elif start_dist_prac.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-2.000000)
    thisExp.nextEntry()
    
    # set up handler to look after randomisation of conditions etc
    dist_practice = data.TrialHandler2(
        name='dist_practice',
        nReps=1.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('conditions.xlsx'), 
        seed=None, 
    )
    thisExp.addLoop(dist_practice)  # add the loop to the experiment
    thisDist_practice = dist_practice.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisDist_practice.rgb)
    if thisDist_practice != None:
        for paramName in thisDist_practice:
            globals()[paramName] = thisDist_practice[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisDist_practice in dist_practice:
        currentLoop = dist_practice
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisDist_practice.rgb)
        if thisDist_practice != None:
            for paramName in thisDist_practice:
                globals()[paramName] = thisDist_practice[paramName]
        
        # --- Prepare to start Routine "dist_trial" ---
        # create an object to store info about Routine dist_trial
        dist_trial = data.Routine(
            name='dist_trial',
            components=[Smileycentral, key_resp_gaze, Smiley, circle, distractorcircle],
        )
        dist_trial.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        Smileycentral.setImage('Smiley_central.png')
        # create starting attributes for key_resp_gaze
        key_resp_gaze.keys = []
        key_resp_gaze.rt = []
        _key_resp_gaze_allKeys = []
        Smiley.setImage(Smiley_file)
        circle.setPos(circlePos)
        distractorcircle.setPos(distractorPos)
        # store start times for dist_trial
        dist_trial.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        dist_trial.tStart = globalClock.getTime(format='float')
        dist_trial.status = STARTED
        thisExp.addData('dist_trial.started', dist_trial.tStart)
        dist_trial.maxDuration = None
        # keep track of which components have finished
        dist_trialComponents = dist_trial.components
        for thisComponent in dist_trial.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "dist_trial" ---
        # if trial has changed, end Routine now
        if isinstance(dist_practice, data.TrialHandler2) and thisDist_practice.thisN != dist_practice.thisTrial.thisN:
            continueRoutine = False
        dist_trial.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *Smileycentral* updates
            
            # if Smileycentral is starting this frame...
            if Smileycentral.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                Smileycentral.frameNStart = frameN  # exact frame index
                Smileycentral.tStart = t  # local t and not account for scr refresh
                Smileycentral.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Smileycentral, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Smileycentral.started')
                # update status
                Smileycentral.status = STARTED
                Smileycentral.setAutoDraw(True)
            
            # if Smileycentral is active this frame...
            if Smileycentral.status == STARTED:
                # update params
                pass
            
            # if Smileycentral is stopping this frame...
            if Smileycentral.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > Smileycentral.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    Smileycentral.tStop = t  # not accounting for scr refresh
                    Smileycentral.tStopRefresh = tThisFlipGlobal  # on global time
                    Smileycentral.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'Smileycentral.stopped')
                    # update status
                    Smileycentral.status = FINISHED
                    Smileycentral.setAutoDraw(False)
            
            # *key_resp_gaze* updates
            waitOnFlip = False
            
            # if key_resp_gaze is starting this frame...
            if key_resp_gaze.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                key_resp_gaze.frameNStart = frameN  # exact frame index
                key_resp_gaze.tStart = t  # local t and not account for scr refresh
                key_resp_gaze.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_gaze, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_gaze.started')
                # update status
                key_resp_gaze.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_gaze.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_gaze.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp_gaze.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_gaze.getKeys(keyList=['2', '3'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_gaze_allKeys.extend(theseKeys)
                if len(_key_resp_gaze_allKeys):
                    key_resp_gaze.keys = _key_resp_gaze_allKeys[-1].name  # just the last key pressed
                    key_resp_gaze.rt = _key_resp_gaze_allKeys[-1].rt
                    key_resp_gaze.duration = _key_resp_gaze_allKeys[-1].duration
                    # was this correct?
                    if (key_resp_gaze.keys == str(correctResponse)) or (key_resp_gaze.keys == correctResponse):
                        key_resp_gaze.corr = 1
                    else:
                        key_resp_gaze.corr = 0
                    # a response ends the routine
                    continueRoutine = False
            
            # *Smiley* updates
            
            # if Smiley is starting this frame...
            if Smiley.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                Smiley.frameNStart = frameN  # exact frame index
                Smiley.tStart = t  # local t and not account for scr refresh
                Smiley.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Smiley, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Smiley.started')
                # update status
                Smiley.status = STARTED
                Smiley.setAutoDraw(True)
            
            # if Smiley is active this frame...
            if Smiley.status == STARTED:
                # update params
                pass
            
            # *circle* updates
            
            # if circle is starting this frame...
            if circle.status == NOT_STARTED and tThisFlip >= 0.7-frameTolerance:
                # keep track of start time/frame for later
                circle.frameNStart = frameN  # exact frame index
                circle.tStart = t  # local t and not account for scr refresh
                circle.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(circle, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'circle.started')
                # update status
                circle.status = STARTED
                circle.setAutoDraw(True)
            
            # if circle is active this frame...
            if circle.status == STARTED:
                # update params
                pass
            
            # if circle is stopping this frame...
            if circle.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > circle.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    circle.tStop = t  # not accounting for scr refresh
                    circle.tStopRefresh = tThisFlipGlobal  # on global time
                    circle.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'circle.stopped')
                    # update status
                    circle.status = FINISHED
                    circle.setAutoDraw(False)
            
            # *distractorcircle* updates
            
            # if distractorcircle is starting this frame...
            if distractorcircle.status == NOT_STARTED and tThisFlip >= 0.7-frameTolerance:
                # keep track of start time/frame for later
                distractorcircle.frameNStart = frameN  # exact frame index
                distractorcircle.tStart = t  # local t and not account for scr refresh
                distractorcircle.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(distractorcircle, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'distractorcircle.started')
                # update status
                distractorcircle.status = STARTED
                distractorcircle.setAutoDraw(True)
            
            # if distractorcircle is active this frame...
            if distractorcircle.status == STARTED:
                # update params
                pass
            
            # if distractorcircle is stopping this frame...
            if distractorcircle.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > distractorcircle.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    distractorcircle.tStop = t  # not accounting for scr refresh
                    distractorcircle.tStopRefresh = tThisFlipGlobal  # on global time
                    distractorcircle.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'distractorcircle.stopped')
                    # update status
                    distractorcircle.status = FINISHED
                    distractorcircle.setAutoDraw(False)
            # Run 'Each Frame' code from dist_code
            if core.getTime() - startTime >= 60:
                if dt == 1:
                    dist_trials.finished = True
                elif dt == 0:
                    dist_practice.finished = True
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                dist_trial.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in dist_trial.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "dist_trial" ---
        for thisComponent in dist_trial.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for dist_trial
        dist_trial.tStop = globalClock.getTime(format='float')
        dist_trial.tStopRefresh = tThisFlipGlobal
        thisExp.addData('dist_trial.stopped', dist_trial.tStop)
        # check responses
        if key_resp_gaze.keys in ['', [], None]:  # No response was made
            key_resp_gaze.keys = None
            # was no response the correct answer?!
            if str(correctResponse).lower() == 'none':
               key_resp_gaze.corr = 1;  # correct non-response
            else:
               key_resp_gaze.corr = 0;  # failed to respond (incorrectly)
        # store data for dist_practice (TrialHandler)
        dist_practice.addData('key_resp_gaze.keys',key_resp_gaze.keys)
        dist_practice.addData('key_resp_gaze.corr', key_resp_gaze.corr)
        if key_resp_gaze.keys != None:  # we had a response
            dist_practice.addData('key_resp_gaze.rt', key_resp_gaze.rt)
            dist_practice.addData('key_resp_gaze.duration', key_resp_gaze.duration)
        # Run 'End Routine' code from dist_code
        fback_text = ""
        rt = ""
        
        if key_resp_gaze.corr == 1:
            rt = "{:.2f} s".format(key_resp_gaze.rt)
            fback_text = "Gut gemacht!\nReaktionszeit: "
            fback_text = fback_text + rt
        else:
            fback_text = "Sie haben den falschen Kopf gedrückt."
        
        # the Routine "dist_trial" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "dist_feedback" ---
        # create an object to store info about Routine dist_feedback
        dist_feedback = data.Routine(
            name='dist_feedback',
            components=[dist_fback_text],
        )
        dist_feedback.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        dist_fback_text.setText(fback_text)
        # store start times for dist_feedback
        dist_feedback.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        dist_feedback.tStart = globalClock.getTime(format='float')
        dist_feedback.status = STARTED
        thisExp.addData('dist_feedback.started', dist_feedback.tStart)
        dist_feedback.maxDuration = None
        # keep track of which components have finished
        dist_feedbackComponents = dist_feedback.components
        for thisComponent in dist_feedback.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "dist_feedback" ---
        # if trial has changed, end Routine now
        if isinstance(dist_practice, data.TrialHandler2) and thisDist_practice.thisN != dist_practice.thisTrial.thisN:
            continueRoutine = False
        dist_feedback.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 3.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *dist_fback_text* updates
            
            # if dist_fback_text is starting this frame...
            if dist_fback_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                dist_fback_text.frameNStart = frameN  # exact frame index
                dist_fback_text.tStart = t  # local t and not account for scr refresh
                dist_fback_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(dist_fback_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'dist_fback_text.started')
                # update status
                dist_fback_text.status = STARTED
                dist_fback_text.setAutoDraw(True)
            
            # if dist_fback_text is active this frame...
            if dist_fback_text.status == STARTED:
                # update params
                pass
            
            # if dist_fback_text is stopping this frame...
            if dist_fback_text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > dist_fback_text.tStartRefresh + 3.0-frameTolerance:
                    # keep track of stop time/frame for later
                    dist_fback_text.tStop = t  # not accounting for scr refresh
                    dist_fback_text.tStopRefresh = tThisFlipGlobal  # on global time
                    dist_fback_text.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'dist_fback_text.stopped')
                    # update status
                    dist_fback_text.status = FINISHED
                    dist_fback_text.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                dist_feedback.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in dist_feedback.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "dist_feedback" ---
        for thisComponent in dist_feedback.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for dist_feedback
        dist_feedback.tStop = globalClock.getTime(format='float')
        dist_feedback.tStopRefresh = tThisFlipGlobal
        thisExp.addData('dist_feedback.stopped', dist_feedback.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if dist_feedback.maxDurationReached:
            routineTimer.addTime(-dist_feedback.maxDuration)
        elif dist_feedback.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-3.000000)
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'dist_practice'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "start_dist" ---
    # create an object to store info about Routine start_dist
    start_dist = data.Routine(
        name='start_dist',
        components=[start_dist_text],
    )
    start_dist.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for start_dist
    start_dist.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    start_dist.tStart = globalClock.getTime(format='float')
    start_dist.status = STARTED
    thisExp.addData('start_dist.started', start_dist.tStart)
    start_dist.maxDuration = None
    # keep track of which components have finished
    start_distComponents = start_dist.components
    for thisComponent in start_dist.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "start_dist" ---
    start_dist.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 3.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *start_dist_text* updates
        
        # if start_dist_text is starting this frame...
        if start_dist_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            start_dist_text.frameNStart = frameN  # exact frame index
            start_dist_text.tStart = t  # local t and not account for scr refresh
            start_dist_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(start_dist_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'start_dist_text.started')
            # update status
            start_dist_text.status = STARTED
            start_dist_text.setAutoDraw(True)
        
        # if start_dist_text is active this frame...
        if start_dist_text.status == STARTED:
            # update params
            pass
        
        # if start_dist_text is stopping this frame...
        if start_dist_text.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > start_dist_text.tStartRefresh + 3.0-frameTolerance:
                # keep track of stop time/frame for later
                start_dist_text.tStop = t  # not accounting for scr refresh
                start_dist_text.tStopRefresh = tThisFlipGlobal  # on global time
                start_dist_text.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'start_dist_text.stopped')
                # update status
                start_dist_text.status = FINISHED
                start_dist_text.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            start_dist.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in start_dist.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "start_dist" ---
    for thisComponent in start_dist.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for start_dist
    start_dist.tStop = globalClock.getTime(format='float')
    start_dist.tStopRefresh = tThisFlipGlobal
    thisExp.addData('start_dist.stopped', start_dist.tStop)
    # Run 'End Routine' code from set_dt
    dt = 1
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if start_dist.maxDurationReached:
        routineTimer.addTime(-start_dist.maxDuration)
    elif start_dist.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-3.000000)
    thisExp.nextEntry()
    
    # set up handler to look after randomisation of conditions etc
    dist_trials = data.TrialHandler2(
        name='dist_trials',
        nReps=1999.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('conditions.xlsx'), 
        seed=None, 
    )
    thisExp.addLoop(dist_trials)  # add the loop to the experiment
    thisDist_trial = dist_trials.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisDist_trial.rgb)
    if thisDist_trial != None:
        for paramName in thisDist_trial:
            globals()[paramName] = thisDist_trial[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisDist_trial in dist_trials:
        currentLoop = dist_trials
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisDist_trial.rgb)
        if thisDist_trial != None:
            for paramName in thisDist_trial:
                globals()[paramName] = thisDist_trial[paramName]
        
        # --- Prepare to start Routine "dist_trial" ---
        # create an object to store info about Routine dist_trial
        dist_trial = data.Routine(
            name='dist_trial',
            components=[Smileycentral, key_resp_gaze, Smiley, circle, distractorcircle],
        )
        dist_trial.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        Smileycentral.setImage('Smiley_central.png')
        # create starting attributes for key_resp_gaze
        key_resp_gaze.keys = []
        key_resp_gaze.rt = []
        _key_resp_gaze_allKeys = []
        Smiley.setImage(Smiley_file)
        circle.setPos(circlePos)
        distractorcircle.setPos(distractorPos)
        # store start times for dist_trial
        dist_trial.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        dist_trial.tStart = globalClock.getTime(format='float')
        dist_trial.status = STARTED
        thisExp.addData('dist_trial.started', dist_trial.tStart)
        dist_trial.maxDuration = None
        # keep track of which components have finished
        dist_trialComponents = dist_trial.components
        for thisComponent in dist_trial.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "dist_trial" ---
        # if trial has changed, end Routine now
        if isinstance(dist_trials, data.TrialHandler2) and thisDist_trial.thisN != dist_trials.thisTrial.thisN:
            continueRoutine = False
        dist_trial.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *Smileycentral* updates
            
            # if Smileycentral is starting this frame...
            if Smileycentral.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                Smileycentral.frameNStart = frameN  # exact frame index
                Smileycentral.tStart = t  # local t and not account for scr refresh
                Smileycentral.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Smileycentral, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Smileycentral.started')
                # update status
                Smileycentral.status = STARTED
                Smileycentral.setAutoDraw(True)
            
            # if Smileycentral is active this frame...
            if Smileycentral.status == STARTED:
                # update params
                pass
            
            # if Smileycentral is stopping this frame...
            if Smileycentral.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > Smileycentral.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    Smileycentral.tStop = t  # not accounting for scr refresh
                    Smileycentral.tStopRefresh = tThisFlipGlobal  # on global time
                    Smileycentral.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'Smileycentral.stopped')
                    # update status
                    Smileycentral.status = FINISHED
                    Smileycentral.setAutoDraw(False)
            
            # *key_resp_gaze* updates
            waitOnFlip = False
            
            # if key_resp_gaze is starting this frame...
            if key_resp_gaze.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                key_resp_gaze.frameNStart = frameN  # exact frame index
                key_resp_gaze.tStart = t  # local t and not account for scr refresh
                key_resp_gaze.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_gaze, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_gaze.started')
                # update status
                key_resp_gaze.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_gaze.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_gaze.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp_gaze.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_gaze.getKeys(keyList=['2', '3'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_gaze_allKeys.extend(theseKeys)
                if len(_key_resp_gaze_allKeys):
                    key_resp_gaze.keys = _key_resp_gaze_allKeys[-1].name  # just the last key pressed
                    key_resp_gaze.rt = _key_resp_gaze_allKeys[-1].rt
                    key_resp_gaze.duration = _key_resp_gaze_allKeys[-1].duration
                    # was this correct?
                    if (key_resp_gaze.keys == str(correctResponse)) or (key_resp_gaze.keys == correctResponse):
                        key_resp_gaze.corr = 1
                    else:
                        key_resp_gaze.corr = 0
                    # a response ends the routine
                    continueRoutine = False
            
            # *Smiley* updates
            
            # if Smiley is starting this frame...
            if Smiley.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                Smiley.frameNStart = frameN  # exact frame index
                Smiley.tStart = t  # local t and not account for scr refresh
                Smiley.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Smiley, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Smiley.started')
                # update status
                Smiley.status = STARTED
                Smiley.setAutoDraw(True)
            
            # if Smiley is active this frame...
            if Smiley.status == STARTED:
                # update params
                pass
            
            # *circle* updates
            
            # if circle is starting this frame...
            if circle.status == NOT_STARTED and tThisFlip >= 0.7-frameTolerance:
                # keep track of start time/frame for later
                circle.frameNStart = frameN  # exact frame index
                circle.tStart = t  # local t and not account for scr refresh
                circle.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(circle, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'circle.started')
                # update status
                circle.status = STARTED
                circle.setAutoDraw(True)
            
            # if circle is active this frame...
            if circle.status == STARTED:
                # update params
                pass
            
            # if circle is stopping this frame...
            if circle.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > circle.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    circle.tStop = t  # not accounting for scr refresh
                    circle.tStopRefresh = tThisFlipGlobal  # on global time
                    circle.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'circle.stopped')
                    # update status
                    circle.status = FINISHED
                    circle.setAutoDraw(False)
            
            # *distractorcircle* updates
            
            # if distractorcircle is starting this frame...
            if distractorcircle.status == NOT_STARTED and tThisFlip >= 0.7-frameTolerance:
                # keep track of start time/frame for later
                distractorcircle.frameNStart = frameN  # exact frame index
                distractorcircle.tStart = t  # local t and not account for scr refresh
                distractorcircle.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(distractorcircle, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'distractorcircle.started')
                # update status
                distractorcircle.status = STARTED
                distractorcircle.setAutoDraw(True)
            
            # if distractorcircle is active this frame...
            if distractorcircle.status == STARTED:
                # update params
                pass
            
            # if distractorcircle is stopping this frame...
            if distractorcircle.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > distractorcircle.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    distractorcircle.tStop = t  # not accounting for scr refresh
                    distractorcircle.tStopRefresh = tThisFlipGlobal  # on global time
                    distractorcircle.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'distractorcircle.stopped')
                    # update status
                    distractorcircle.status = FINISHED
                    distractorcircle.setAutoDraw(False)
            # Run 'Each Frame' code from dist_code
            if core.getTime() - startTime >= 60:
                if dt == 1:
                    dist_trials.finished = True
                elif dt == 0:
                    dist_practice.finished = True
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                dist_trial.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in dist_trial.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "dist_trial" ---
        for thisComponent in dist_trial.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for dist_trial
        dist_trial.tStop = globalClock.getTime(format='float')
        dist_trial.tStopRefresh = tThisFlipGlobal
        thisExp.addData('dist_trial.stopped', dist_trial.tStop)
        # check responses
        if key_resp_gaze.keys in ['', [], None]:  # No response was made
            key_resp_gaze.keys = None
            # was no response the correct answer?!
            if str(correctResponse).lower() == 'none':
               key_resp_gaze.corr = 1;  # correct non-response
            else:
               key_resp_gaze.corr = 0;  # failed to respond (incorrectly)
        # store data for dist_trials (TrialHandler)
        dist_trials.addData('key_resp_gaze.keys',key_resp_gaze.keys)
        dist_trials.addData('key_resp_gaze.corr', key_resp_gaze.corr)
        if key_resp_gaze.keys != None:  # we had a response
            dist_trials.addData('key_resp_gaze.rt', key_resp_gaze.rt)
            dist_trials.addData('key_resp_gaze.duration', key_resp_gaze.duration)
        # Run 'End Routine' code from dist_code
        fback_text = ""
        rt = ""
        
        if key_resp_gaze.corr == 1:
            rt = "{:.2f} s".format(key_resp_gaze.rt)
            fback_text = "Gut gemacht!\nReaktionszeit: "
            fback_text = fback_text + rt
        else:
            fback_text = "Sie haben den falschen Kopf gedrückt."
        
        # the Routine "dist_trial" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
    # completed 1999.0 repeats of 'dist_trials'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "instructions_recall1" ---
    # create an object to store info about Routine instructions_recall1
    instructions_recall1 = data.Routine(
        name='instructions_recall1',
        components=[recall_instructions1_text, recall_instructions1_key],
    )
    instructions_recall1.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for recall_instructions1_key
    recall_instructions1_key.keys = []
    recall_instructions1_key.rt = []
    _recall_instructions1_key_allKeys = []
    # allowedKeys looks like a variable, so make sure it exists locally
    if 'all_keys' in globals():
        all_keys = globals()['all_keys']
    # store start times for instructions_recall1
    instructions_recall1.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instructions_recall1.tStart = globalClock.getTime(format='float')
    instructions_recall1.status = STARTED
    thisExp.addData('instructions_recall1.started', instructions_recall1.tStart)
    instructions_recall1.maxDuration = None
    # keep track of which components have finished
    instructions_recall1Components = instructions_recall1.components
    for thisComponent in instructions_recall1.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instructions_recall1" ---
    instructions_recall1.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *recall_instructions1_text* updates
        
        # if recall_instructions1_text is starting this frame...
        if recall_instructions1_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            recall_instructions1_text.frameNStart = frameN  # exact frame index
            recall_instructions1_text.tStart = t  # local t and not account for scr refresh
            recall_instructions1_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(recall_instructions1_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'recall_instructions1_text.started')
            # update status
            recall_instructions1_text.status = STARTED
            recall_instructions1_text.setAutoDraw(True)
        
        # if recall_instructions1_text is active this frame...
        if recall_instructions1_text.status == STARTED:
            # update params
            pass
        
        # *recall_instructions1_key* updates
        waitOnFlip = False
        
        # if recall_instructions1_key is starting this frame...
        if recall_instructions1_key.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            recall_instructions1_key.frameNStart = frameN  # exact frame index
            recall_instructions1_key.tStart = t  # local t and not account for scr refresh
            recall_instructions1_key.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(recall_instructions1_key, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'recall_instructions1_key.started')
            # update status
            recall_instructions1_key.status = STARTED
            # allowed keys looks like a variable named `all_keys`
            if not type(all_keys) in [list, tuple, np.ndarray]:
                if not isinstance(all_keys, str):
                    all_keys = str(all_keys)
                elif not ',' in all_keys:
                    all_keys = (all_keys,)
                else:
                    all_keys = eval(all_keys)
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(recall_instructions1_key.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(recall_instructions1_key.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if recall_instructions1_key.status == STARTED and not waitOnFlip:
            theseKeys = recall_instructions1_key.getKeys(keyList=list(all_keys), ignoreKeys=["escape"], waitRelease=False)
            _recall_instructions1_key_allKeys.extend(theseKeys)
            if len(_recall_instructions1_key_allKeys):
                recall_instructions1_key.keys = _recall_instructions1_key_allKeys[-1].name  # just the last key pressed
                recall_instructions1_key.rt = _recall_instructions1_key_allKeys[-1].rt
                recall_instructions1_key.duration = _recall_instructions1_key_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            instructions_recall1.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instructions_recall1.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instructions_recall1" ---
    for thisComponent in instructions_recall1.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instructions_recall1
    instructions_recall1.tStop = globalClock.getTime(format='float')
    instructions_recall1.tStopRefresh = tThisFlipGlobal
    thisExp.addData('instructions_recall1.stopped', instructions_recall1.tStop)
    # check responses
    if recall_instructions1_key.keys in ['', [], None]:  # No response was made
        recall_instructions1_key.keys = None
    thisExp.addData('recall_instructions1_key.keys',recall_instructions1_key.keys)
    if recall_instructions1_key.keys != None:  # we had a response
        thisExp.addData('recall_instructions1_key.rt', recall_instructions1_key.rt)
        thisExp.addData('recall_instructions1_key.duration', recall_instructions1_key.duration)
    thisExp.nextEntry()
    # the Routine "instructions_recall1" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "instructions_recall2" ---
    # create an object to store info about Routine instructions_recall2
    instructions_recall2 = data.Routine(
        name='instructions_recall2',
        components=[recall_instructions2_text, recall_instructions2_key],
    )
    instructions_recall2.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for recall_instructions2_key
    recall_instructions2_key.keys = []
    recall_instructions2_key.rt = []
    _recall_instructions2_key_allKeys = []
    # allowedKeys looks like a variable, so make sure it exists locally
    if 'all_keys' in globals():
        all_keys = globals()['all_keys']
    # store start times for instructions_recall2
    instructions_recall2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instructions_recall2.tStart = globalClock.getTime(format='float')
    instructions_recall2.status = STARTED
    thisExp.addData('instructions_recall2.started', instructions_recall2.tStart)
    instructions_recall2.maxDuration = None
    # keep track of which components have finished
    instructions_recall2Components = instructions_recall2.components
    for thisComponent in instructions_recall2.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instructions_recall2" ---
    instructions_recall2.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *recall_instructions2_text* updates
        
        # if recall_instructions2_text is starting this frame...
        if recall_instructions2_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            recall_instructions2_text.frameNStart = frameN  # exact frame index
            recall_instructions2_text.tStart = t  # local t and not account for scr refresh
            recall_instructions2_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(recall_instructions2_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'recall_instructions2_text.started')
            # update status
            recall_instructions2_text.status = STARTED
            recall_instructions2_text.setAutoDraw(True)
        
        # if recall_instructions2_text is active this frame...
        if recall_instructions2_text.status == STARTED:
            # update params
            pass
        
        # *recall_instructions2_key* updates
        waitOnFlip = False
        
        # if recall_instructions2_key is starting this frame...
        if recall_instructions2_key.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            recall_instructions2_key.frameNStart = frameN  # exact frame index
            recall_instructions2_key.tStart = t  # local t and not account for scr refresh
            recall_instructions2_key.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(recall_instructions2_key, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'recall_instructions2_key.started')
            # update status
            recall_instructions2_key.status = STARTED
            # allowed keys looks like a variable named `all_keys`
            if not type(all_keys) in [list, tuple, np.ndarray]:
                if not isinstance(all_keys, str):
                    all_keys = str(all_keys)
                elif not ',' in all_keys:
                    all_keys = (all_keys,)
                else:
                    all_keys = eval(all_keys)
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(recall_instructions2_key.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(recall_instructions2_key.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if recall_instructions2_key.status == STARTED and not waitOnFlip:
            theseKeys = recall_instructions2_key.getKeys(keyList=list(all_keys), ignoreKeys=["escape"], waitRelease=False)
            _recall_instructions2_key_allKeys.extend(theseKeys)
            if len(_recall_instructions2_key_allKeys):
                recall_instructions2_key.keys = _recall_instructions2_key_allKeys[-1].name  # just the last key pressed
                recall_instructions2_key.rt = _recall_instructions2_key_allKeys[-1].rt
                recall_instructions2_key.duration = _recall_instructions2_key_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            instructions_recall2.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instructions_recall2.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instructions_recall2" ---
    for thisComponent in instructions_recall2.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instructions_recall2
    instructions_recall2.tStop = globalClock.getTime(format='float')
    instructions_recall2.tStopRefresh = tThisFlipGlobal
    thisExp.addData('instructions_recall2.stopped', instructions_recall2.tStop)
    # check responses
    if recall_instructions2_key.keys in ['', [], None]:  # No response was made
        recall_instructions2_key.keys = None
    thisExp.addData('recall_instructions2_key.keys',recall_instructions2_key.keys)
    if recall_instructions2_key.keys != None:  # we had a response
        thisExp.addData('recall_instructions2_key.rt', recall_instructions2_key.rt)
        thisExp.addData('recall_instructions2_key.duration', recall_instructions2_key.duration)
    thisExp.nextEntry()
    # the Routine "instructions_recall2" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    recall_practice = data.TrialHandler2(
        name='recall_practice',
        nReps=1.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('practice.xlsx'), 
        seed=None, 
    )
    thisExp.addLoop(recall_practice)  # add the loop to the experiment
    thisRecall_practice = recall_practice.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisRecall_practice.rgb)
    if thisRecall_practice != None:
        for paramName in thisRecall_practice:
            globals()[paramName] = thisRecall_practice[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisRecall_practice in recall_practice:
        currentLoop = recall_practice
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisRecall_practice.rgb)
        if thisRecall_practice != None:
            for paramName in thisRecall_practice:
                globals()[paramName] = thisRecall_practice[paramName]
        
        # --- Prepare to start Routine "iti_recall_practice" ---
        # create an object to store info about Routine iti_recall_practice
        iti_recall_practice = data.Routine(
            name='iti_recall_practice',
            components=[iti_recall_practice_cross],
        )
        iti_recall_practice.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for iti_recall_practice
        iti_recall_practice.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        iti_recall_practice.tStart = globalClock.getTime(format='float')
        iti_recall_practice.status = STARTED
        thisExp.addData('iti_recall_practice.started', iti_recall_practice.tStart)
        iti_recall_practice.maxDuration = None
        # keep track of which components have finished
        iti_recall_practiceComponents = iti_recall_practice.components
        for thisComponent in iti_recall_practice.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "iti_recall_practice" ---
        # if trial has changed, end Routine now
        if isinstance(recall_practice, data.TrialHandler2) and thisRecall_practice.thisN != recall_practice.thisTrial.thisN:
            continueRoutine = False
        iti_recall_practice.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 2.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *iti_recall_practice_cross* updates
            
            # if iti_recall_practice_cross is starting this frame...
            if iti_recall_practice_cross.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                iti_recall_practice_cross.frameNStart = frameN  # exact frame index
                iti_recall_practice_cross.tStart = t  # local t and not account for scr refresh
                iti_recall_practice_cross.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(iti_recall_practice_cross, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'iti_recall_practice_cross.started')
                # update status
                iti_recall_practice_cross.status = STARTED
                iti_recall_practice_cross.setAutoDraw(True)
            
            # if iti_recall_practice_cross is active this frame...
            if iti_recall_practice_cross.status == STARTED:
                # update params
                pass
            
            # if iti_recall_practice_cross is stopping this frame...
            if iti_recall_practice_cross.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > iti_recall_practice_cross.tStartRefresh + 2.0-frameTolerance:
                    # keep track of stop time/frame for later
                    iti_recall_practice_cross.tStop = t  # not accounting for scr refresh
                    iti_recall_practice_cross.tStopRefresh = tThisFlipGlobal  # on global time
                    iti_recall_practice_cross.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'iti_recall_practice_cross.stopped')
                    # update status
                    iti_recall_practice_cross.status = FINISHED
                    iti_recall_practice_cross.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                iti_recall_practice.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in iti_recall_practice.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "iti_recall_practice" ---
        for thisComponent in iti_recall_practice.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for iti_recall_practice
        iti_recall_practice.tStop = globalClock.getTime(format='float')
        iti_recall_practice.tStopRefresh = tThisFlipGlobal
        thisExp.addData('iti_recall_practice.stopped', iti_recall_practice.tStop)
        # Run 'End Routine' code from set_recall_practice_stim
        this_cue = cue
        this_target = target
        run_counter = 0
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if iti_recall_practice.maxDurationReached:
            routineTimer.addTime(-iti_recall_practice.maxDuration)
        elif iti_recall_practice.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-2.000000)
        
        # --- Prepare to start Routine "cued_recall" ---
        # create an object to store info about Routine cued_recall
        cued_recall = data.Routine(
            name='cued_recall',
            components=[dash_stim_recall, qmark_target, end_cued_recall, cue_stim_recall],
        )
        cued_recall.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from save_recall_trial_type
        thisExp.addData('trial_type', cue_types[this_cue])
        thisExp.addData('Cue', this_cue)
        thisExp.addData('Target', this_target)
        thisExp.addData('RunNr', run_counter)
        
        # create starting attributes for end_cued_recall
        end_cued_recall.keys = []
        end_cued_recall.rt = []
        _end_cued_recall_allKeys = []
        cue_stim_recall.setText(this_cue)
        # store start times for cued_recall
        cued_recall.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        cued_recall.tStart = globalClock.getTime(format='float')
        cued_recall.status = STARTED
        thisExp.addData('cued_recall.started', cued_recall.tStart)
        cued_recall.maxDuration = None
        # keep track of which components have finished
        cued_recallComponents = cued_recall.components
        for thisComponent in cued_recall.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "cued_recall" ---
        # if trial has changed, end Routine now
        if isinstance(recall_practice, data.TrialHandler2) and thisRecall_practice.thisN != recall_practice.thisTrial.thisN:
            continueRoutine = False
        cued_recall.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 3.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *dash_stim_recall* updates
            
            # if dash_stim_recall is starting this frame...
            if dash_stim_recall.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                dash_stim_recall.frameNStart = frameN  # exact frame index
                dash_stim_recall.tStart = t  # local t and not account for scr refresh
                dash_stim_recall.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(dash_stim_recall, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'dash_stim_recall.started')
                # update status
                dash_stim_recall.status = STARTED
                dash_stim_recall.setAutoDraw(True)
            
            # if dash_stim_recall is active this frame...
            if dash_stim_recall.status == STARTED:
                # update params
                pass
            
            # if dash_stim_recall is stopping this frame...
            if dash_stim_recall.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > dash_stim_recall.tStartRefresh + 3.0-frameTolerance:
                    # keep track of stop time/frame for later
                    dash_stim_recall.tStop = t  # not accounting for scr refresh
                    dash_stim_recall.tStopRefresh = tThisFlipGlobal  # on global time
                    dash_stim_recall.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'dash_stim_recall.stopped')
                    # update status
                    dash_stim_recall.status = FINISHED
                    dash_stim_recall.setAutoDraw(False)
            
            # *qmark_target* updates
            
            # if qmark_target is starting this frame...
            if qmark_target.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                qmark_target.frameNStart = frameN  # exact frame index
                qmark_target.tStart = t  # local t and not account for scr refresh
                qmark_target.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(qmark_target, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'qmark_target.started')
                # update status
                qmark_target.status = STARTED
                qmark_target.setAutoDraw(True)
            
            # if qmark_target is active this frame...
            if qmark_target.status == STARTED:
                # update params
                pass
            
            # if qmark_target is stopping this frame...
            if qmark_target.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > qmark_target.tStartRefresh + 3.0-frameTolerance:
                    # keep track of stop time/frame for later
                    qmark_target.tStop = t  # not accounting for scr refresh
                    qmark_target.tStopRefresh = tThisFlipGlobal  # on global time
                    qmark_target.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'qmark_target.stopped')
                    # update status
                    qmark_target.status = FINISHED
                    qmark_target.setAutoDraw(False)
            
            # *end_cued_recall* updates
            waitOnFlip = False
            
            # if end_cued_recall is starting this frame...
            if end_cued_recall.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                end_cued_recall.frameNStart = frameN  # exact frame index
                end_cued_recall.tStart = t  # local t and not account for scr refresh
                end_cued_recall.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(end_cued_recall, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'end_cued_recall.started')
                # update status
                end_cued_recall.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(end_cued_recall.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(end_cued_recall.clearEvents, eventType='keyboard')  # clear events on next screen flip
            
            # if end_cued_recall is stopping this frame...
            if end_cued_recall.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > end_cued_recall.tStartRefresh + 3.0-frameTolerance:
                    # keep track of stop time/frame for later
                    end_cued_recall.tStop = t  # not accounting for scr refresh
                    end_cued_recall.tStopRefresh = tThisFlipGlobal  # on global time
                    end_cued_recall.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'end_cued_recall.stopped')
                    # update status
                    end_cued_recall.status = FINISHED
                    end_cued_recall.status = FINISHED
            if end_cued_recall.status == STARTED and not waitOnFlip:
                theseKeys = end_cued_recall.getKeys(keyList=['p'], ignoreKeys=["escape"], waitRelease=False)
                _end_cued_recall_allKeys.extend(theseKeys)
                if len(_end_cued_recall_allKeys):
                    end_cued_recall.keys = _end_cued_recall_allKeys[-1].name  # just the last key pressed
                    end_cued_recall.rt = _end_cued_recall_allKeys[-1].rt
                    end_cued_recall.duration = _end_cued_recall_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # *cue_stim_recall* updates
            
            # if cue_stim_recall is starting this frame...
            if cue_stim_recall.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                cue_stim_recall.frameNStart = frameN  # exact frame index
                cue_stim_recall.tStart = t  # local t and not account for scr refresh
                cue_stim_recall.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(cue_stim_recall, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'cue_stim_recall.started')
                # update status
                cue_stim_recall.status = STARTED
                cue_stim_recall.setAutoDraw(True)
            
            # if cue_stim_recall is active this frame...
            if cue_stim_recall.status == STARTED:
                # update params
                pass
            
            # if cue_stim_recall is stopping this frame...
            if cue_stim_recall.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > cue_stim_recall.tStartRefresh + 3.0-frameTolerance:
                    # keep track of stop time/frame for later
                    cue_stim_recall.tStop = t  # not accounting for scr refresh
                    cue_stim_recall.tStopRefresh = tThisFlipGlobal  # on global time
                    cue_stim_recall.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'cue_stim_recall.stopped')
                    # update status
                    cue_stim_recall.status = FINISHED
                    cue_stim_recall.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                cued_recall.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in cued_recall.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "cued_recall" ---
        for thisComponent in cued_recall.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for cued_recall
        cued_recall.tStop = globalClock.getTime(format='float')
        cued_recall.tStopRefresh = tThisFlipGlobal
        thisExp.addData('cued_recall.stopped', cued_recall.tStop)
        # check responses
        if end_cued_recall.keys in ['', [], None]:  # No response was made
            end_cued_recall.keys = None
        recall_practice.addData('end_cued_recall.keys',end_cued_recall.keys)
        if end_cued_recall.keys != None:  # we had a response
            recall_practice.addData('end_cued_recall.rt', end_cued_recall.rt)
            recall_practice.addData('end_cued_recall.duration', end_cued_recall.duration)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if cued_recall.maxDurationReached:
            routineTimer.addTime(-cued_recall.maxDuration)
        elif cued_recall.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-3.000000)
        
        # --- Prepare to start Routine "recall_response" ---
        # create an object to store info about Routine recall_response
        recall_response = data.Routine(
            name='recall_response',
            components=[dash_stim_recall_resp, cue_stim_recall_resp, recall_stim_resp, recall_question, recall_reached],
        )
        recall_response.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        cue_stim_recall_resp.setText(this_cue)
        # create starting attributes for recall_reached
        recall_reached.keys = []
        recall_reached.rt = []
        _recall_reached_allKeys = []
        # allowedKeys looks like a variable, so make sure it exists locally
        if 'all_keys' in globals():
            all_keys = globals()['all_keys']
        # store start times for recall_response
        recall_response.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        recall_response.tStart = globalClock.getTime(format='float')
        recall_response.status = STARTED
        thisExp.addData('recall_response.started', recall_response.tStart)
        recall_response.maxDuration = None
        # keep track of which components have finished
        recall_responseComponents = recall_response.components
        for thisComponent in recall_response.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "recall_response" ---
        # if trial has changed, end Routine now
        if isinstance(recall_practice, data.TrialHandler2) and thisRecall_practice.thisN != recall_practice.thisTrial.thisN:
            continueRoutine = False
        recall_response.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.5:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *dash_stim_recall_resp* updates
            
            # if dash_stim_recall_resp is starting this frame...
            if dash_stim_recall_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                dash_stim_recall_resp.frameNStart = frameN  # exact frame index
                dash_stim_recall_resp.tStart = t  # local t and not account for scr refresh
                dash_stim_recall_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(dash_stim_recall_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'dash_stim_recall_resp.started')
                # update status
                dash_stim_recall_resp.status = STARTED
                dash_stim_recall_resp.setAutoDraw(True)
            
            # if dash_stim_recall_resp is active this frame...
            if dash_stim_recall_resp.status == STARTED:
                # update params
                pass
            
            # if dash_stim_recall_resp is stopping this frame...
            if dash_stim_recall_resp.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > dash_stim_recall_resp.tStartRefresh + 1.5-frameTolerance:
                    # keep track of stop time/frame for later
                    dash_stim_recall_resp.tStop = t  # not accounting for scr refresh
                    dash_stim_recall_resp.tStopRefresh = tThisFlipGlobal  # on global time
                    dash_stim_recall_resp.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'dash_stim_recall_resp.stopped')
                    # update status
                    dash_stim_recall_resp.status = FINISHED
                    dash_stim_recall_resp.setAutoDraw(False)
            
            # *cue_stim_recall_resp* updates
            
            # if cue_stim_recall_resp is starting this frame...
            if cue_stim_recall_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                cue_stim_recall_resp.frameNStart = frameN  # exact frame index
                cue_stim_recall_resp.tStart = t  # local t and not account for scr refresh
                cue_stim_recall_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(cue_stim_recall_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'cue_stim_recall_resp.started')
                # update status
                cue_stim_recall_resp.status = STARTED
                cue_stim_recall_resp.setAutoDraw(True)
            
            # if cue_stim_recall_resp is active this frame...
            if cue_stim_recall_resp.status == STARTED:
                # update params
                pass
            
            # if cue_stim_recall_resp is stopping this frame...
            if cue_stim_recall_resp.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > cue_stim_recall_resp.tStartRefresh + 1.5-frameTolerance:
                    # keep track of stop time/frame for later
                    cue_stim_recall_resp.tStop = t  # not accounting for scr refresh
                    cue_stim_recall_resp.tStopRefresh = tThisFlipGlobal  # on global time
                    cue_stim_recall_resp.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'cue_stim_recall_resp.stopped')
                    # update status
                    cue_stim_recall_resp.status = FINISHED
                    cue_stim_recall_resp.setAutoDraw(False)
            
            # *recall_stim_resp* updates
            
            # if recall_stim_resp is starting this frame...
            if recall_stim_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                recall_stim_resp.frameNStart = frameN  # exact frame index
                recall_stim_resp.tStart = t  # local t and not account for scr refresh
                recall_stim_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(recall_stim_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'recall_stim_resp.started')
                # update status
                recall_stim_resp.status = STARTED
                recall_stim_resp.setAutoDraw(True)
            
            # if recall_stim_resp is active this frame...
            if recall_stim_resp.status == STARTED:
                # update params
                pass
            
            # if recall_stim_resp is stopping this frame...
            if recall_stim_resp.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > recall_stim_resp.tStartRefresh + 1.5-frameTolerance:
                    # keep track of stop time/frame for later
                    recall_stim_resp.tStop = t  # not accounting for scr refresh
                    recall_stim_resp.tStopRefresh = tThisFlipGlobal  # on global time
                    recall_stim_resp.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'recall_stim_resp.stopped')
                    # update status
                    recall_stim_resp.status = FINISHED
                    recall_stim_resp.setAutoDraw(False)
            
            # *recall_question* updates
            
            # if recall_question is starting this frame...
            if recall_question.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                recall_question.frameNStart = frameN  # exact frame index
                recall_question.tStart = t  # local t and not account for scr refresh
                recall_question.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(recall_question, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'recall_question.started')
                # update status
                recall_question.status = STARTED
                recall_question.setAutoDraw(True)
            
            # if recall_question is active this frame...
            if recall_question.status == STARTED:
                # update params
                pass
            
            # if recall_question is stopping this frame...
            if recall_question.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > recall_question.tStartRefresh + 1.5-frameTolerance:
                    # keep track of stop time/frame for later
                    recall_question.tStop = t  # not accounting for scr refresh
                    recall_question.tStopRefresh = tThisFlipGlobal  # on global time
                    recall_question.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'recall_question.stopped')
                    # update status
                    recall_question.status = FINISHED
                    recall_question.setAutoDraw(False)
            
            # *recall_reached* updates
            waitOnFlip = False
            
            # if recall_reached is starting this frame...
            if recall_reached.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                recall_reached.frameNStart = frameN  # exact frame index
                recall_reached.tStart = t  # local t and not account for scr refresh
                recall_reached.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(recall_reached, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'recall_reached.started')
                # update status
                recall_reached.status = STARTED
                # allowed keys looks like a variable named `all_keys`
                if not type(all_keys) in [list, tuple, np.ndarray]:
                    if not isinstance(all_keys, str):
                        all_keys = str(all_keys)
                    elif not ',' in all_keys:
                        all_keys = (all_keys,)
                    else:
                        all_keys = eval(all_keys)
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(recall_reached.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(recall_reached.clearEvents, eventType='keyboard')  # clear events on next screen flip
            
            # if recall_reached is stopping this frame...
            if recall_reached.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > recall_reached.tStartRefresh + 1.5-frameTolerance:
                    # keep track of stop time/frame for later
                    recall_reached.tStop = t  # not accounting for scr refresh
                    recall_reached.tStopRefresh = tThisFlipGlobal  # on global time
                    recall_reached.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'recall_reached.stopped')
                    # update status
                    recall_reached.status = FINISHED
                    recall_reached.status = FINISHED
            if recall_reached.status == STARTED and not waitOnFlip:
                theseKeys = recall_reached.getKeys(keyList=list(all_keys), ignoreKeys=["escape"], waitRelease=False)
                _recall_reached_allKeys.extend(theseKeys)
                if len(_recall_reached_allKeys):
                    recall_reached.keys = _recall_reached_allKeys[-1].name  # just the last key pressed
                    recall_reached.rt = _recall_reached_allKeys[-1].rt
                    recall_reached.duration = _recall_reached_allKeys[-1].duration
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                recall_response.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in recall_response.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "recall_response" ---
        for thisComponent in recall_response.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for recall_response
        recall_response.tStop = globalClock.getTime(format='float')
        recall_response.tStopRefresh = tThisFlipGlobal
        thisExp.addData('recall_response.stopped', recall_response.tStop)
        # check responses
        if recall_reached.keys in ['', [], None]:  # No response was made
            recall_reached.keys = None
        recall_practice.addData('recall_reached.keys',recall_reached.keys)
        if recall_reached.keys != None:  # we had a response
            recall_practice.addData('recall_reached.rt', recall_reached.rt)
            recall_practice.addData('recall_reached.duration', recall_reached.duration)
        # Run 'End Routine' code from skip_recall_select
        skip_letters = 0
        trial_dur = letter_dur
        if recall_reached.keys == succ_recall_key:
            trial_dur = letter_dur
        else:
            trial_dur = 0.00
            skip_letters = 1
        
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if recall_response.maxDurationReached:
            routineTimer.addTime(-recall_response.maxDuration)
        elif recall_response.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.500000)
        
        # --- Prepare to start Routine "recall_select" ---
        # create an object to store info about Routine recall_select
        recall_select = data.Routine(
            name='recall_select',
            components=[dash_stim_recall_select, cue_stim_recall_select, recall_stim_select, recall_question_select, recall_selection],
        )
        recall_select.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from letter_choice
        recall_select_text = ""
        
        random_letters = rnd.sample("abcdefghijklmnoprstuvz", k=3)
        target_letter = this_target[-1]
        while target_letter in random_letters:
            random_letters = rnd.sample("abcdefghijklmnoprstuvz", k=3)
        
        
        random_letters.append(target_letter)
        
        letters = rnd.sample(random_letters, len(random_letters))
        thisExp.addData('letter_options', letters)
        
        recall_select_text += "1) " + letters[0] + "        "
        recall_select_text += "2) " + letters[1] + "        "
        recall_select_text += "3) " + letters[2] + "        "
        recall_select_text += "4) " + letters[3] 
        
        cue_stim_recall_select.setText(this_cue)
        recall_question_select.setText(recall_select_text)
        # create starting attributes for recall_selection
        recall_selection.keys = []
        recall_selection.rt = []
        _recall_selection_allKeys = []
        # allowedKeys looks like a variable, so make sure it exists locally
        if 'all_keys' in globals():
            all_keys = globals()['all_keys']
        # store start times for recall_select
        recall_select.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        recall_select.tStart = globalClock.getTime(format='float')
        recall_select.status = STARTED
        thisExp.addData('recall_select.started', recall_select.tStart)
        recall_select.maxDuration = None
        # skip Routine recall_select if its 'Skip if' condition is True
        recall_select.skipped = continueRoutine and not (skip_letters >= 1)
        continueRoutine = recall_select.skipped
        # keep track of which components have finished
        recall_selectComponents = recall_select.components
        for thisComponent in recall_select.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "recall_select" ---
        # if trial has changed, end Routine now
        if isinstance(recall_practice, data.TrialHandler2) and thisRecall_practice.thisN != recall_practice.thisTrial.thisN:
            continueRoutine = False
        recall_select.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *dash_stim_recall_select* updates
            
            # if dash_stim_recall_select is starting this frame...
            if dash_stim_recall_select.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                dash_stim_recall_select.frameNStart = frameN  # exact frame index
                dash_stim_recall_select.tStart = t  # local t and not account for scr refresh
                dash_stim_recall_select.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(dash_stim_recall_select, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'dash_stim_recall_select.started')
                # update status
                dash_stim_recall_select.status = STARTED
                dash_stim_recall_select.setAutoDraw(True)
            
            # if dash_stim_recall_select is active this frame...
            if dash_stim_recall_select.status == STARTED:
                # update params
                pass
            
            # if dash_stim_recall_select is stopping this frame...
            if dash_stim_recall_select.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > dash_stim_recall_select.tStartRefresh + trial_dur-frameTolerance:
                    # keep track of stop time/frame for later
                    dash_stim_recall_select.tStop = t  # not accounting for scr refresh
                    dash_stim_recall_select.tStopRefresh = tThisFlipGlobal  # on global time
                    dash_stim_recall_select.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'dash_stim_recall_select.stopped')
                    # update status
                    dash_stim_recall_select.status = FINISHED
                    dash_stim_recall_select.setAutoDraw(False)
            
            # *cue_stim_recall_select* updates
            
            # if cue_stim_recall_select is starting this frame...
            if cue_stim_recall_select.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                cue_stim_recall_select.frameNStart = frameN  # exact frame index
                cue_stim_recall_select.tStart = t  # local t and not account for scr refresh
                cue_stim_recall_select.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(cue_stim_recall_select, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'cue_stim_recall_select.started')
                # update status
                cue_stim_recall_select.status = STARTED
                cue_stim_recall_select.setAutoDraw(True)
            
            # if cue_stim_recall_select is active this frame...
            if cue_stim_recall_select.status == STARTED:
                # update params
                pass
            
            # if cue_stim_recall_select is stopping this frame...
            if cue_stim_recall_select.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > cue_stim_recall_select.tStartRefresh + trial_dur-frameTolerance:
                    # keep track of stop time/frame for later
                    cue_stim_recall_select.tStop = t  # not accounting for scr refresh
                    cue_stim_recall_select.tStopRefresh = tThisFlipGlobal  # on global time
                    cue_stim_recall_select.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'cue_stim_recall_select.stopped')
                    # update status
                    cue_stim_recall_select.status = FINISHED
                    cue_stim_recall_select.setAutoDraw(False)
            
            # *recall_stim_select* updates
            
            # if recall_stim_select is starting this frame...
            if recall_stim_select.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                recall_stim_select.frameNStart = frameN  # exact frame index
                recall_stim_select.tStart = t  # local t and not account for scr refresh
                recall_stim_select.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(recall_stim_select, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'recall_stim_select.started')
                # update status
                recall_stim_select.status = STARTED
                recall_stim_select.setAutoDraw(True)
            
            # if recall_stim_select is active this frame...
            if recall_stim_select.status == STARTED:
                # update params
                pass
            
            # if recall_stim_select is stopping this frame...
            if recall_stim_select.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > recall_stim_select.tStartRefresh + trial_dur-frameTolerance:
                    # keep track of stop time/frame for later
                    recall_stim_select.tStop = t  # not accounting for scr refresh
                    recall_stim_select.tStopRefresh = tThisFlipGlobal  # on global time
                    recall_stim_select.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'recall_stim_select.stopped')
                    # update status
                    recall_stim_select.status = FINISHED
                    recall_stim_select.setAutoDraw(False)
            
            # *recall_question_select* updates
            
            # if recall_question_select is starting this frame...
            if recall_question_select.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                recall_question_select.frameNStart = frameN  # exact frame index
                recall_question_select.tStart = t  # local t and not account for scr refresh
                recall_question_select.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(recall_question_select, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'recall_question_select.started')
                # update status
                recall_question_select.status = STARTED
                recall_question_select.setAutoDraw(True)
            
            # if recall_question_select is active this frame...
            if recall_question_select.status == STARTED:
                # update params
                pass
            
            # if recall_question_select is stopping this frame...
            if recall_question_select.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > recall_question_select.tStartRefresh + trial_dur-frameTolerance:
                    # keep track of stop time/frame for later
                    recall_question_select.tStop = t  # not accounting for scr refresh
                    recall_question_select.tStopRefresh = tThisFlipGlobal  # on global time
                    recall_question_select.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'recall_question_select.stopped')
                    # update status
                    recall_question_select.status = FINISHED
                    recall_question_select.setAutoDraw(False)
            
            # *recall_selection* updates
            waitOnFlip = False
            
            # if recall_selection is starting this frame...
            if recall_selection.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                recall_selection.frameNStart = frameN  # exact frame index
                recall_selection.tStart = t  # local t and not account for scr refresh
                recall_selection.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(recall_selection, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'recall_selection.started')
                # update status
                recall_selection.status = STARTED
                # allowed keys looks like a variable named `all_keys`
                if not type(all_keys) in [list, tuple, np.ndarray]:
                    if not isinstance(all_keys, str):
                        all_keys = str(all_keys)
                    elif not ',' in all_keys:
                        all_keys = (all_keys,)
                    else:
                        all_keys = eval(all_keys)
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(recall_selection.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(recall_selection.clearEvents, eventType='keyboard')  # clear events on next screen flip
            
            # if recall_selection is stopping this frame...
            if recall_selection.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > recall_selection.tStartRefresh + trial_dur-frameTolerance:
                    # keep track of stop time/frame for later
                    recall_selection.tStop = t  # not accounting for scr refresh
                    recall_selection.tStopRefresh = tThisFlipGlobal  # on global time
                    recall_selection.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'recall_selection.stopped')
                    # update status
                    recall_selection.status = FINISHED
                    recall_selection.status = FINISHED
            if recall_selection.status == STARTED and not waitOnFlip:
                theseKeys = recall_selection.getKeys(keyList=list(all_keys), ignoreKeys=["escape"], waitRelease=False)
                _recall_selection_allKeys.extend(theseKeys)
                if len(_recall_selection_allKeys):
                    recall_selection.keys = _recall_selection_allKeys[-1].name  # just the last key pressed
                    recall_selection.rt = _recall_selection_allKeys[-1].rt
                    recall_selection.duration = _recall_selection_allKeys[-1].duration
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                recall_select.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in recall_select.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "recall_select" ---
        for thisComponent in recall_select.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for recall_select
        recall_select.tStop = globalClock.getTime(format='float')
        recall_select.tStopRefresh = tThisFlipGlobal
        thisExp.addData('recall_select.stopped', recall_select.tStop)
        # Run 'End Routine' code from letter_choice
        this_choice = ""
        correct_choice = 0
        if skip_letters  <= 1:
            if len(recall_selection.keys) > 0:
                this_key = int(recall_selection.keys[0]) - 2
                this_choice = letters[this_key]
        
                if this_choice == target_letter:
                    correct_choice = 1
                    if (run_counter == 1) or (run_counter == 2):
                        ncorrect = ncorrect + 1
                        
                        if cue_types[this_cue] == "Read":
                            correct_read = correct_read + 1
                        elif cue_types[this_cue] == "Guess":
                            correct_guess = correct_guess + 1
        
        thisExp.addData('correct_choice', correct_choice)
        # check responses
        if recall_selection.keys in ['', [], None]:  # No response was made
            recall_selection.keys = None
        recall_practice.addData('recall_selection.keys',recall_selection.keys)
        if recall_selection.keys != None:  # we had a response
            recall_practice.addData('recall_selection.rt', recall_selection.rt)
            recall_practice.addData('recall_selection.duration', recall_selection.duration)
        # the Routine "recall_select" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "recall_feedback" ---
        # create an object to store info about Routine recall_feedback
        recall_feedback = data.Routine(
            name='recall_feedback',
            components=[dash_stim_feedback, cue_stim_feedback, target_stim_feedback, end_feedback, recall_feedback1, recall_feedback2],
        )
        recall_feedback.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        cue_stim_feedback.setText(cue)
        target_stim_feedback.setText(target)
        # create starting attributes for end_feedback
        end_feedback.keys = []
        end_feedback.rt = []
        _end_feedback_allKeys = []
        recall_feedback2.setText(this_target[-1])
        # store start times for recall_feedback
        recall_feedback.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        recall_feedback.tStart = globalClock.getTime(format='float')
        recall_feedback.status = STARTED
        thisExp.addData('recall_feedback.started', recall_feedback.tStart)
        recall_feedback.maxDuration = None
        # keep track of which components have finished
        recall_feedbackComponents = recall_feedback.components
        for thisComponent in recall_feedback.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "recall_feedback" ---
        # if trial has changed, end Routine now
        if isinstance(recall_practice, data.TrialHandler2) and thisRecall_practice.thisN != recall_practice.thisTrial.thisN:
            continueRoutine = False
        recall_feedback.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 3.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *dash_stim_feedback* updates
            
            # if dash_stim_feedback is starting this frame...
            if dash_stim_feedback.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                dash_stim_feedback.frameNStart = frameN  # exact frame index
                dash_stim_feedback.tStart = t  # local t and not account for scr refresh
                dash_stim_feedback.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(dash_stim_feedback, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'dash_stim_feedback.started')
                # update status
                dash_stim_feedback.status = STARTED
                dash_stim_feedback.setAutoDraw(True)
            
            # if dash_stim_feedback is active this frame...
            if dash_stim_feedback.status == STARTED:
                # update params
                pass
            
            # if dash_stim_feedback is stopping this frame...
            if dash_stim_feedback.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > dash_stim_feedback.tStartRefresh + 3.0-frameTolerance:
                    # keep track of stop time/frame for later
                    dash_stim_feedback.tStop = t  # not accounting for scr refresh
                    dash_stim_feedback.tStopRefresh = tThisFlipGlobal  # on global time
                    dash_stim_feedback.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'dash_stim_feedback.stopped')
                    # update status
                    dash_stim_feedback.status = FINISHED
                    dash_stim_feedback.setAutoDraw(False)
            
            # *cue_stim_feedback* updates
            
            # if cue_stim_feedback is starting this frame...
            if cue_stim_feedback.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                cue_stim_feedback.frameNStart = frameN  # exact frame index
                cue_stim_feedback.tStart = t  # local t and not account for scr refresh
                cue_stim_feedback.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(cue_stim_feedback, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'cue_stim_feedback.started')
                # update status
                cue_stim_feedback.status = STARTED
                cue_stim_feedback.setAutoDraw(True)
            
            # if cue_stim_feedback is active this frame...
            if cue_stim_feedback.status == STARTED:
                # update params
                pass
            
            # if cue_stim_feedback is stopping this frame...
            if cue_stim_feedback.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > cue_stim_feedback.tStartRefresh + 3.0-frameTolerance:
                    # keep track of stop time/frame for later
                    cue_stim_feedback.tStop = t  # not accounting for scr refresh
                    cue_stim_feedback.tStopRefresh = tThisFlipGlobal  # on global time
                    cue_stim_feedback.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'cue_stim_feedback.stopped')
                    # update status
                    cue_stim_feedback.status = FINISHED
                    cue_stim_feedback.setAutoDraw(False)
            
            # *target_stim_feedback* updates
            
            # if target_stim_feedback is starting this frame...
            if target_stim_feedback.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                target_stim_feedback.frameNStart = frameN  # exact frame index
                target_stim_feedback.tStart = t  # local t and not account for scr refresh
                target_stim_feedback.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(target_stim_feedback, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'target_stim_feedback.started')
                # update status
                target_stim_feedback.status = STARTED
                target_stim_feedback.setAutoDraw(True)
            
            # if target_stim_feedback is active this frame...
            if target_stim_feedback.status == STARTED:
                # update params
                pass
            
            # if target_stim_feedback is stopping this frame...
            if target_stim_feedback.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > target_stim_feedback.tStartRefresh + 3.0-frameTolerance:
                    # keep track of stop time/frame for later
                    target_stim_feedback.tStop = t  # not accounting for scr refresh
                    target_stim_feedback.tStopRefresh = tThisFlipGlobal  # on global time
                    target_stim_feedback.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'target_stim_feedback.stopped')
                    # update status
                    target_stim_feedback.status = FINISHED
                    target_stim_feedback.setAutoDraw(False)
            
            # *end_feedback* updates
            waitOnFlip = False
            
            # if end_feedback is starting this frame...
            if end_feedback.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                end_feedback.frameNStart = frameN  # exact frame index
                end_feedback.tStart = t  # local t and not account for scr refresh
                end_feedback.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(end_feedback, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'end_feedback.started')
                # update status
                end_feedback.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(end_feedback.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(end_feedback.clearEvents, eventType='keyboard')  # clear events on next screen flip
            
            # if end_feedback is stopping this frame...
            if end_feedback.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > end_feedback.tStartRefresh + 3.00-frameTolerance:
                    # keep track of stop time/frame for later
                    end_feedback.tStop = t  # not accounting for scr refresh
                    end_feedback.tStopRefresh = tThisFlipGlobal  # on global time
                    end_feedback.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'end_feedback.stopped')
                    # update status
                    end_feedback.status = FINISHED
                    end_feedback.status = FINISHED
            if end_feedback.status == STARTED and not waitOnFlip:
                theseKeys = end_feedback.getKeys(keyList=['p'], ignoreKeys=["escape"], waitRelease=False)
                _end_feedback_allKeys.extend(theseKeys)
                if len(_end_feedback_allKeys):
                    end_feedback.keys = _end_feedback_allKeys[-1].name  # just the last key pressed
                    end_feedback.rt = _end_feedback_allKeys[-1].rt
                    end_feedback.duration = _end_feedback_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # *recall_feedback1* updates
            
            # if recall_feedback1 is starting this frame...
            if recall_feedback1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                recall_feedback1.frameNStart = frameN  # exact frame index
                recall_feedback1.tStart = t  # local t and not account for scr refresh
                recall_feedback1.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(recall_feedback1, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'recall_feedback1.started')
                # update status
                recall_feedback1.status = STARTED
                recall_feedback1.setAutoDraw(True)
            
            # if recall_feedback1 is active this frame...
            if recall_feedback1.status == STARTED:
                # update params
                pass
            
            # if recall_feedback1 is stopping this frame...
            if recall_feedback1.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > recall_feedback1.tStartRefresh + 3.0-frameTolerance:
                    # keep track of stop time/frame for later
                    recall_feedback1.tStop = t  # not accounting for scr refresh
                    recall_feedback1.tStopRefresh = tThisFlipGlobal  # on global time
                    recall_feedback1.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'recall_feedback1.stopped')
                    # update status
                    recall_feedback1.status = FINISHED
                    recall_feedback1.setAutoDraw(False)
            
            # *recall_feedback2* updates
            
            # if recall_feedback2 is starting this frame...
            if recall_feedback2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                recall_feedback2.frameNStart = frameN  # exact frame index
                recall_feedback2.tStart = t  # local t and not account for scr refresh
                recall_feedback2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(recall_feedback2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'recall_feedback2.started')
                # update status
                recall_feedback2.status = STARTED
                recall_feedback2.setAutoDraw(True)
            
            # if recall_feedback2 is active this frame...
            if recall_feedback2.status == STARTED:
                # update params
                pass
            
            # if recall_feedback2 is stopping this frame...
            if recall_feedback2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > recall_feedback2.tStartRefresh + 3.0-frameTolerance:
                    # keep track of stop time/frame for later
                    recall_feedback2.tStop = t  # not accounting for scr refresh
                    recall_feedback2.tStopRefresh = tThisFlipGlobal  # on global time
                    recall_feedback2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'recall_feedback2.stopped')
                    # update status
                    recall_feedback2.status = FINISHED
                    recall_feedback2.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                recall_feedback.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in recall_feedback.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "recall_feedback" ---
        for thisComponent in recall_feedback.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for recall_feedback
        recall_feedback.tStop = globalClock.getTime(format='float')
        recall_feedback.tStopRefresh = tThisFlipGlobal
        thisExp.addData('recall_feedback.stopped', recall_feedback.tStop)
        # check responses
        if end_feedback.keys in ['', [], None]:  # No response was made
            end_feedback.keys = None
        recall_practice.addData('end_feedback.keys',end_feedback.keys)
        if end_feedback.keys != None:  # we had a response
            recall_practice.addData('end_feedback.rt', end_feedback.rt)
            recall_practice.addData('end_feedback.duration', end_feedback.duration)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if recall_feedback.maxDurationReached:
            routineTimer.addTime(-recall_feedback.maxDuration)
        elif recall_feedback.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-3.000000)
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'recall_practice'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # set up handler to look after randomisation of conditions etc
    recall_words_loop = data.TrialHandler2(
        name='recall_words_loop',
        nReps=1.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('cue_target_pairs.xlsx'), 
        seed=None, 
    )
    thisExp.addLoop(recall_words_loop)  # add the loop to the experiment
    thisRecall_words_loop = recall_words_loop.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisRecall_words_loop.rgb)
    if thisRecall_words_loop != None:
        for paramName in thisRecall_words_loop:
            globals()[paramName] = thisRecall_words_loop[paramName]
    
    for thisRecall_words_loop in recall_words_loop:
        currentLoop = recall_words_loop
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        # abbreviate parameter names if possible (e.g. rgb = thisRecall_words_loop.rgb)
        if thisRecall_words_loop != None:
            for paramName in thisRecall_words_loop:
                globals()[paramName] = thisRecall_words_loop[paramName]
        
        # --- Prepare to start Routine "load_word_pairs" ---
        # create an object to store info about Routine load_word_pairs
        load_word_pairs = data.Routine(
            name='load_word_pairs',
            components=[],
        )
        load_word_pairs.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from load_stimuli
        cue_list.append(cue)
        target_list.append(target)
        
        # store start times for load_word_pairs
        load_word_pairs.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        load_word_pairs.tStart = globalClock.getTime(format='float')
        load_word_pairs.status = STARTED
        thisExp.addData('load_word_pairs.started', load_word_pairs.tStart)
        load_word_pairs.maxDuration = None
        # keep track of which components have finished
        load_word_pairsComponents = load_word_pairs.components
        for thisComponent in load_word_pairs.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "load_word_pairs" ---
        # if trial has changed, end Routine now
        if isinstance(recall_words_loop, data.TrialHandler2) and thisRecall_words_loop.thisN != recall_words_loop.thisTrial.thisN:
            continueRoutine = False
        load_word_pairs.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                load_word_pairs.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in load_word_pairs.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "load_word_pairs" ---
        for thisComponent in load_word_pairs.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for load_word_pairs
        load_word_pairs.tStop = globalClock.getTime(format='float')
        load_word_pairs.tStopRefresh = tThisFlipGlobal
        thisExp.addData('load_word_pairs.stopped', load_word_pairs.tStop)
        # the Routine "load_word_pairs" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
    # completed 1.0 repeats of 'recall_words_loop'
    
    
    # --- Prepare to start Routine "start_task" ---
    # create an object to store info about Routine start_task
    start_task = data.Routine(
        name='start_task',
        components=[start_task_text, start_key],
    )
    start_task.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for start_key
    start_key.keys = []
    start_key.rt = []
    _start_key_allKeys = []
    # allowedKeys looks like a variable, so make sure it exists locally
    if 'all_keys' in globals():
        all_keys = globals()['all_keys']
    # Run 'Begin Routine' code from set_up_run_counter
    run_counter = 1
    # store start times for start_task
    start_task.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    start_task.tStart = globalClock.getTime(format='float')
    start_task.status = STARTED
    thisExp.addData('start_task.started', start_task.tStart)
    start_task.maxDuration = None
    # keep track of which components have finished
    start_taskComponents = start_task.components
    for thisComponent in start_task.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "start_task" ---
    start_task.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *start_task_text* updates
        
        # if start_task_text is starting this frame...
        if start_task_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            start_task_text.frameNStart = frameN  # exact frame index
            start_task_text.tStart = t  # local t and not account for scr refresh
            start_task_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(start_task_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'start_task_text.started')
            # update status
            start_task_text.status = STARTED
            start_task_text.setAutoDraw(True)
        
        # if start_task_text is active this frame...
        if start_task_text.status == STARTED:
            # update params
            pass
        
        # *start_key* updates
        waitOnFlip = False
        
        # if start_key is starting this frame...
        if start_key.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            start_key.frameNStart = frameN  # exact frame index
            start_key.tStart = t  # local t and not account for scr refresh
            start_key.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(start_key, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'start_key.started')
            # update status
            start_key.status = STARTED
            # allowed keys looks like a variable named `all_keys`
            if not type(all_keys) in [list, tuple, np.ndarray]:
                if not isinstance(all_keys, str):
                    all_keys = str(all_keys)
                elif not ',' in all_keys:
                    all_keys = (all_keys,)
                else:
                    all_keys = eval(all_keys)
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(start_key.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(start_key.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if start_key.status == STARTED and not waitOnFlip:
            theseKeys = start_key.getKeys(keyList=list(all_keys), ignoreKeys=["escape"], waitRelease=False)
            _start_key_allKeys.extend(theseKeys)
            if len(_start_key_allKeys):
                start_key.keys = _start_key_allKeys[-1].name  # just the last key pressed
                start_key.rt = _start_key_allKeys[-1].rt
                start_key.duration = _start_key_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            start_task.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in start_task.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "start_task" ---
    for thisComponent in start_task.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for start_task
    start_task.tStop = globalClock.getTime(format='float')
    start_task.tStopRefresh = tThisFlipGlobal
    thisExp.addData('start_task.stopped', start_task.tStop)
    # check responses
    if start_key.keys in ['', [], None]:  # No response was made
        start_key.keys = None
    thisExp.addData('start_key.keys',start_key.keys)
    if start_key.keys != None:  # we had a response
        thisExp.addData('start_key.rt', start_key.rt)
        thisExp.addData('start_key.duration', start_key.duration)
    thisExp.nextEntry()
    # the Routine "start_task" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    recall_block = data.TrialHandler2(
        name='recall_block',
        nReps=2.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=[None], 
        seed=None, 
    )
    thisExp.addLoop(recall_block)  # add the loop to the experiment
    thisRecall_block = recall_block.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisRecall_block.rgb)
    if thisRecall_block != None:
        for paramName in thisRecall_block:
            globals()[paramName] = thisRecall_block[paramName]
    
    for thisRecall_block in recall_block:
        currentLoop = recall_block
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        # abbreviate parameter names if possible (e.g. rgb = thisRecall_block.rgb)
        if thisRecall_block != None:
            for paramName in thisRecall_block:
                globals()[paramName] = thisRecall_block[paramName]
        
        # set up handler to look after randomisation of conditions etc
        iti_recall_loop = data.TrialHandler2(
            name='iti_recall_loop',
            nReps=1.0, 
            method='random', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=data.importConditions('iti_randomization.csv'), 
            seed=None, 
        )
        thisExp.addLoop(iti_recall_loop)  # add the loop to the experiment
        thisIti_recall_loop = iti_recall_loop.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisIti_recall_loop.rgb)
        if thisIti_recall_loop != None:
            for paramName in thisIti_recall_loop:
                globals()[paramName] = thisIti_recall_loop[paramName]
        
        for thisIti_recall_loop in iti_recall_loop:
            currentLoop = iti_recall_loop
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            # abbreviate parameter names if possible (e.g. rgb = thisIti_recall_loop.rgb)
            if thisIti_recall_loop != None:
                for paramName in thisIti_recall_loop:
                    globals()[paramName] = thisIti_recall_loop[paramName]
            
            # --- Prepare to start Routine "load_iti" ---
            # create an object to store info about Routine load_iti
            load_iti = data.Routine(
                name='load_iti',
                components=[],
            )
            load_iti.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from load_iti_code
            iti_list.append(iti)
            
            # store start times for load_iti
            load_iti.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            load_iti.tStart = globalClock.getTime(format='float')
            load_iti.status = STARTED
            thisExp.addData('load_iti.started', load_iti.tStart)
            load_iti.maxDuration = None
            # keep track of which components have finished
            load_itiComponents = load_iti.components
            for thisComponent in load_iti.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "load_iti" ---
            # if trial has changed, end Routine now
            if isinstance(iti_recall_loop, data.TrialHandler2) and thisIti_recall_loop.thisN != iti_recall_loop.thisTrial.thisN:
                continueRoutine = False
            load_iti.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    load_iti.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in load_iti.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "load_iti" ---
            for thisComponent in load_iti.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for load_iti
            load_iti.tStop = globalClock.getTime(format='float')
            load_iti.tStopRefresh = tThisFlipGlobal
            thisExp.addData('load_iti.stopped', load_iti.tStop)
            # the Routine "load_iti" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
        # completed 1.0 repeats of 'iti_recall_loop'
        
        
        # --- Prepare to start Routine "prep_scanner" ---
        # create an object to store info about Routine prep_scanner
        prep_scanner = data.Routine(
            name='prep_scanner',
            components=[scanner_ready_press, ready_set_text],
        )
        prep_scanner.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # create starting attributes for scanner_ready_press
        scanner_ready_press.keys = []
        scanner_ready_press.rt = []
        _scanner_ready_press_allKeys = []
        # store start times for prep_scanner
        prep_scanner.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        prep_scanner.tStart = globalClock.getTime(format='float')
        prep_scanner.status = STARTED
        thisExp.addData('prep_scanner.started', prep_scanner.tStart)
        prep_scanner.maxDuration = None
        # keep track of which components have finished
        prep_scannerComponents = prep_scanner.components
        for thisComponent in prep_scanner.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "prep_scanner" ---
        # if trial has changed, end Routine now
        if isinstance(recall_block, data.TrialHandler2) and thisRecall_block.thisN != recall_block.thisTrial.thisN:
            continueRoutine = False
        prep_scanner.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *scanner_ready_press* updates
            waitOnFlip = False
            
            # if scanner_ready_press is starting this frame...
            if scanner_ready_press.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                scanner_ready_press.frameNStart = frameN  # exact frame index
                scanner_ready_press.tStart = t  # local t and not account for scr refresh
                scanner_ready_press.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(scanner_ready_press, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'scanner_ready_press.started')
                # update status
                scanner_ready_press.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(scanner_ready_press.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(scanner_ready_press.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if scanner_ready_press.status == STARTED and not waitOnFlip:
                theseKeys = scanner_ready_press.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _scanner_ready_press_allKeys.extend(theseKeys)
                if len(_scanner_ready_press_allKeys):
                    scanner_ready_press.keys = _scanner_ready_press_allKeys[-1].name  # just the last key pressed
                    scanner_ready_press.rt = _scanner_ready_press_allKeys[-1].rt
                    scanner_ready_press.duration = _scanner_ready_press_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # *ready_set_text* updates
            
            # if ready_set_text is starting this frame...
            if ready_set_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                ready_set_text.frameNStart = frameN  # exact frame index
                ready_set_text.tStart = t  # local t and not account for scr refresh
                ready_set_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(ready_set_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'ready_set_text.started')
                # update status
                ready_set_text.status = STARTED
                ready_set_text.setAutoDraw(True)
            
            # if ready_set_text is active this frame...
            if ready_set_text.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                prep_scanner.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in prep_scanner.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "prep_scanner" ---
        for thisComponent in prep_scanner.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for prep_scanner
        prep_scanner.tStop = globalClock.getTime(format='float')
        prep_scanner.tStopRefresh = tThisFlipGlobal
        thisExp.addData('prep_scanner.stopped', prep_scanner.tStop)
        # check responses
        if scanner_ready_press.keys in ['', [], None]:  # No response was made
            scanner_ready_press.keys = None
        recall_block.addData('scanner_ready_press.keys',scanner_ready_press.keys)
        if scanner_ready_press.keys != None:  # we had a response
            recall_block.addData('scanner_ready_press.rt', scanner_ready_press.rt)
            recall_block.addData('scanner_ready_press.duration', scanner_ready_press.duration)
        # the Routine "prep_scanner" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "wait_for_trigger" ---
        # create an object to store info about Routine wait_for_trigger
        wait_for_trigger = data.Routine(
            name='wait_for_trigger',
            components=[starting_soon, skip_trigger],
        )
        wait_for_trigger.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # create starting attributes for skip_trigger
        skip_trigger.keys = []
        skip_trigger.rt = []
        _skip_trigger_allKeys = []
        # store start times for wait_for_trigger
        wait_for_trigger.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        wait_for_trigger.tStart = globalClock.getTime(format='float')
        wait_for_trigger.status = STARTED
        thisExp.addData('wait_for_trigger.started', wait_for_trigger.tStart)
        wait_for_trigger.maxDuration = None
        # keep track of which components have finished
        wait_for_triggerComponents = wait_for_trigger.components
        for thisComponent in wait_for_trigger.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "wait_for_trigger" ---
        # if trial has changed, end Routine now
        if isinstance(recall_block, data.TrialHandler2) and thisRecall_block.thisN != recall_block.thisTrial.thisN:
            continueRoutine = False
        wait_for_trigger.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # Run 'Each Frame' code from catch_trigger
            if expInfo["MRI"] == "1":
                if port.readPin(pinNumber) > 0:
                    continueRoutine = False #A trigger was detected, so move on
            
            # *starting_soon* updates
            
            # if starting_soon is starting this frame...
            if starting_soon.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                starting_soon.frameNStart = frameN  # exact frame index
                starting_soon.tStart = t  # local t and not account for scr refresh
                starting_soon.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(starting_soon, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'starting_soon.started')
                # update status
                starting_soon.status = STARTED
                starting_soon.setAutoDraw(True)
            
            # if starting_soon is active this frame...
            if starting_soon.status == STARTED:
                # update params
                pass
            
            # *skip_trigger* updates
            waitOnFlip = False
            
            # if skip_trigger is starting this frame...
            if skip_trigger.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                skip_trigger.frameNStart = frameN  # exact frame index
                skip_trigger.tStart = t  # local t and not account for scr refresh
                skip_trigger.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(skip_trigger, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'skip_trigger.started')
                # update status
                skip_trigger.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(skip_trigger.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(skip_trigger.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if skip_trigger.status == STARTED and not waitOnFlip:
                theseKeys = skip_trigger.getKeys(keyList=['p'], ignoreKeys=["escape"], waitRelease=False)
                _skip_trigger_allKeys.extend(theseKeys)
                if len(_skip_trigger_allKeys):
                    skip_trigger.keys = _skip_trigger_allKeys[-1].name  # just the last key pressed
                    skip_trigger.rt = _skip_trigger_allKeys[-1].rt
                    skip_trigger.duration = _skip_trigger_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                wait_for_trigger.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in wait_for_trigger.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "wait_for_trigger" ---
        for thisComponent in wait_for_trigger.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for wait_for_trigger
        wait_for_trigger.tStop = globalClock.getTime(format='float')
        wait_for_trigger.tStopRefresh = tThisFlipGlobal
        thisExp.addData('wait_for_trigger.stopped', wait_for_trigger.tStop)
        # check responses
        if skip_trigger.keys in ['', [], None]:  # No response was made
            skip_trigger.keys = None
        recall_block.addData('skip_trigger.keys',skip_trigger.keys)
        if skip_trigger.keys != None:  # we had a response
            recall_block.addData('skip_trigger.rt', skip_trigger.rt)
            recall_block.addData('skip_trigger.duration', skip_trigger.duration)
        # the Routine "wait_for_trigger" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "blank" ---
        # create an object to store info about Routine blank
        blank = data.Routine(
            name='blank',
            components=[begin_end_run_cross, sound_1, sound_2],
        )
        blank.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        sound_1.setSound('bell.wav', secs=1.2, hamming=True)
        sound_1.setVolume(1.0, log=False)
        sound_1.seek(0)
        sound_2.setSound('end_call.wav', secs=1, hamming=True)
        sound_2.setVolume(1.0, log=False)
        sound_2.seek(0)
        # store start times for blank
        blank.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        blank.tStart = globalClock.getTime(format='float')
        blank.status = STARTED
        thisExp.addData('blank.started', blank.tStart)
        blank.maxDuration = None
        # keep track of which components have finished
        blankComponents = blank.components
        for thisComponent in blank.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "blank" ---
        # if trial has changed, end Routine now
        if isinstance(recall_block, data.TrialHandler2) and thisRecall_block.thisN != recall_block.thisTrial.thisN:
            continueRoutine = False
        blank.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 12.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *begin_end_run_cross* updates
            
            # if begin_end_run_cross is starting this frame...
            if begin_end_run_cross.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                begin_end_run_cross.frameNStart = frameN  # exact frame index
                begin_end_run_cross.tStart = t  # local t and not account for scr refresh
                begin_end_run_cross.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(begin_end_run_cross, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'begin_end_run_cross.started')
                # update status
                begin_end_run_cross.status = STARTED
                begin_end_run_cross.setAutoDraw(True)
            
            # if begin_end_run_cross is active this frame...
            if begin_end_run_cross.status == STARTED:
                # update params
                pass
            
            # if begin_end_run_cross is stopping this frame...
            if begin_end_run_cross.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > begin_end_run_cross.tStartRefresh + 12.0-frameTolerance:
                    # keep track of stop time/frame for later
                    begin_end_run_cross.tStop = t  # not accounting for scr refresh
                    begin_end_run_cross.tStopRefresh = tThisFlipGlobal  # on global time
                    begin_end_run_cross.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'begin_end_run_cross.stopped')
                    # update status
                    begin_end_run_cross.status = FINISHED
                    begin_end_run_cross.setAutoDraw(False)
            
            # *sound_1* updates
            
            # if sound_1 is starting this frame...
            if sound_1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                sound_1.frameNStart = frameN  # exact frame index
                sound_1.tStart = t  # local t and not account for scr refresh
                sound_1.tStartRefresh = tThisFlipGlobal  # on global time
                # add timestamp to datafile
                thisExp.addData('sound_1.started', tThisFlipGlobal)
                # update status
                sound_1.status = STARTED
                sound_1.play(when=win)  # sync with win flip
            
            # if sound_1 is stopping this frame...
            if sound_1.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > sound_1.tStartRefresh + 1.2-frameTolerance or sound_1.isFinished:
                    # keep track of stop time/frame for later
                    sound_1.tStop = t  # not accounting for scr refresh
                    sound_1.tStopRefresh = tThisFlipGlobal  # on global time
                    sound_1.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'sound_1.stopped')
                    # update status
                    sound_1.status = FINISHED
                    sound_1.stop()
            
            # *sound_2* updates
            
            # if sound_2 is starting this frame...
            if sound_2.status == NOT_STARTED and tThisFlip >= 10.0-frameTolerance:
                # keep track of start time/frame for later
                sound_2.frameNStart = frameN  # exact frame index
                sound_2.tStart = t  # local t and not account for scr refresh
                sound_2.tStartRefresh = tThisFlipGlobal  # on global time
                # add timestamp to datafile
                thisExp.addData('sound_2.started', tThisFlipGlobal)
                # update status
                sound_2.status = STARTED
                sound_2.play(when=win)  # sync with win flip
            
            # if sound_2 is stopping this frame...
            if sound_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > sound_2.tStartRefresh + 1-frameTolerance or sound_2.isFinished:
                    # keep track of stop time/frame for later
                    sound_2.tStop = t  # not accounting for scr refresh
                    sound_2.tStopRefresh = tThisFlipGlobal  # on global time
                    sound_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'sound_2.stopped')
                    # update status
                    sound_2.status = FINISHED
                    sound_2.stop()
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[sound_1, sound_2]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                blank.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in blank.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "blank" ---
        for thisComponent in blank.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for blank
        blank.tStop = globalClock.getTime(format='float')
        blank.tStopRefresh = tThisFlipGlobal
        thisExp.addData('blank.stopped', blank.tStop)
        sound_1.pause()  # ensure sound has stopped at end of Routine
        sound_2.pause()  # ensure sound has stopped at end of Routine
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if blank.maxDurationReached:
            routineTimer.addTime(-blank.maxDuration)
        elif blank.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-12.000000)
        
        # set up handler to look after randomisation of conditions etc
        test_trials = data.TrialHandler2(
            name='test_trials',
            nReps=num_trials, 
            method='random', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=[None], 
            seed=None, 
        )
        thisExp.addLoop(test_trials)  # add the loop to the experiment
        thisTest_trial = test_trials.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisTest_trial.rgb)
        if thisTest_trial != None:
            for paramName in thisTest_trial:
                globals()[paramName] = thisTest_trial[paramName]
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        for thisTest_trial in test_trials:
            currentLoop = test_trials
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # abbreviate parameter names if possible (e.g. rgb = thisTest_trial.rgb)
            if thisTest_trial != None:
                for paramName in thisTest_trial:
                    globals()[paramName] = thisTest_trial[paramName]
            
            # --- Prepare to start Routine "setup_recall_trial" ---
            # create an object to store info about Routine setup_recall_trial
            setup_recall_trial = data.Routine(
                name='setup_recall_trial',
                components=[],
            )
            setup_recall_trial.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from set_iti_stim_recall
            this_iti = iti_list.pop()
            this_cue = cue_list.pop()
            this_target = target_list.pop()
            
            if len(cue_list) == 40:
                print('Last recall trial in run')
            elif len(cue_list) == 0:
                print('Last recall trial')
            # store start times for setup_recall_trial
            setup_recall_trial.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            setup_recall_trial.tStart = globalClock.getTime(format='float')
            setup_recall_trial.status = STARTED
            thisExp.addData('setup_recall_trial.started', setup_recall_trial.tStart)
            setup_recall_trial.maxDuration = None
            # keep track of which components have finished
            setup_recall_trialComponents = setup_recall_trial.components
            for thisComponent in setup_recall_trial.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "setup_recall_trial" ---
            # if trial has changed, end Routine now
            if isinstance(test_trials, data.TrialHandler2) and thisTest_trial.thisN != test_trials.thisTrial.thisN:
                continueRoutine = False
            setup_recall_trial.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    setup_recall_trial.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in setup_recall_trial.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "setup_recall_trial" ---
            for thisComponent in setup_recall_trial.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for setup_recall_trial
            setup_recall_trial.tStop = globalClock.getTime(format='float')
            setup_recall_trial.tStopRefresh = tThisFlipGlobal
            thisExp.addData('setup_recall_trial.stopped', setup_recall_trial.tStop)
            # the Routine "setup_recall_trial" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "cued_recall" ---
            # create an object to store info about Routine cued_recall
            cued_recall = data.Routine(
                name='cued_recall',
                components=[dash_stim_recall, qmark_target, end_cued_recall, cue_stim_recall],
            )
            cued_recall.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from save_recall_trial_type
            thisExp.addData('trial_type', cue_types[this_cue])
            thisExp.addData('Cue', this_cue)
            thisExp.addData('Target', this_target)
            thisExp.addData('RunNr', run_counter)
            
            # create starting attributes for end_cued_recall
            end_cued_recall.keys = []
            end_cued_recall.rt = []
            _end_cued_recall_allKeys = []
            cue_stim_recall.setText(this_cue)
            # store start times for cued_recall
            cued_recall.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            cued_recall.tStart = globalClock.getTime(format='float')
            cued_recall.status = STARTED
            thisExp.addData('cued_recall.started', cued_recall.tStart)
            cued_recall.maxDuration = None
            # keep track of which components have finished
            cued_recallComponents = cued_recall.components
            for thisComponent in cued_recall.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "cued_recall" ---
            # if trial has changed, end Routine now
            if isinstance(test_trials, data.TrialHandler2) and thisTest_trial.thisN != test_trials.thisTrial.thisN:
                continueRoutine = False
            cued_recall.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 3.0:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *dash_stim_recall* updates
                
                # if dash_stim_recall is starting this frame...
                if dash_stim_recall.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    dash_stim_recall.frameNStart = frameN  # exact frame index
                    dash_stim_recall.tStart = t  # local t and not account for scr refresh
                    dash_stim_recall.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(dash_stim_recall, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'dash_stim_recall.started')
                    # update status
                    dash_stim_recall.status = STARTED
                    dash_stim_recall.setAutoDraw(True)
                
                # if dash_stim_recall is active this frame...
                if dash_stim_recall.status == STARTED:
                    # update params
                    pass
                
                # if dash_stim_recall is stopping this frame...
                if dash_stim_recall.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > dash_stim_recall.tStartRefresh + 3.0-frameTolerance:
                        # keep track of stop time/frame for later
                        dash_stim_recall.tStop = t  # not accounting for scr refresh
                        dash_stim_recall.tStopRefresh = tThisFlipGlobal  # on global time
                        dash_stim_recall.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'dash_stim_recall.stopped')
                        # update status
                        dash_stim_recall.status = FINISHED
                        dash_stim_recall.setAutoDraw(False)
                
                # *qmark_target* updates
                
                # if qmark_target is starting this frame...
                if qmark_target.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    qmark_target.frameNStart = frameN  # exact frame index
                    qmark_target.tStart = t  # local t and not account for scr refresh
                    qmark_target.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(qmark_target, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'qmark_target.started')
                    # update status
                    qmark_target.status = STARTED
                    qmark_target.setAutoDraw(True)
                
                # if qmark_target is active this frame...
                if qmark_target.status == STARTED:
                    # update params
                    pass
                
                # if qmark_target is stopping this frame...
                if qmark_target.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > qmark_target.tStartRefresh + 3.0-frameTolerance:
                        # keep track of stop time/frame for later
                        qmark_target.tStop = t  # not accounting for scr refresh
                        qmark_target.tStopRefresh = tThisFlipGlobal  # on global time
                        qmark_target.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'qmark_target.stopped')
                        # update status
                        qmark_target.status = FINISHED
                        qmark_target.setAutoDraw(False)
                
                # *end_cued_recall* updates
                waitOnFlip = False
                
                # if end_cued_recall is starting this frame...
                if end_cued_recall.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    end_cued_recall.frameNStart = frameN  # exact frame index
                    end_cued_recall.tStart = t  # local t and not account for scr refresh
                    end_cued_recall.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(end_cued_recall, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'end_cued_recall.started')
                    # update status
                    end_cued_recall.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(end_cued_recall.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(end_cued_recall.clearEvents, eventType='keyboard')  # clear events on next screen flip
                
                # if end_cued_recall is stopping this frame...
                if end_cued_recall.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > end_cued_recall.tStartRefresh + 3.0-frameTolerance:
                        # keep track of stop time/frame for later
                        end_cued_recall.tStop = t  # not accounting for scr refresh
                        end_cued_recall.tStopRefresh = tThisFlipGlobal  # on global time
                        end_cued_recall.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'end_cued_recall.stopped')
                        # update status
                        end_cued_recall.status = FINISHED
                        end_cued_recall.status = FINISHED
                if end_cued_recall.status == STARTED and not waitOnFlip:
                    theseKeys = end_cued_recall.getKeys(keyList=['p'], ignoreKeys=["escape"], waitRelease=False)
                    _end_cued_recall_allKeys.extend(theseKeys)
                    if len(_end_cued_recall_allKeys):
                        end_cued_recall.keys = _end_cued_recall_allKeys[-1].name  # just the last key pressed
                        end_cued_recall.rt = _end_cued_recall_allKeys[-1].rt
                        end_cued_recall.duration = _end_cued_recall_allKeys[-1].duration
                        # a response ends the routine
                        continueRoutine = False
                
                # *cue_stim_recall* updates
                
                # if cue_stim_recall is starting this frame...
                if cue_stim_recall.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    cue_stim_recall.frameNStart = frameN  # exact frame index
                    cue_stim_recall.tStart = t  # local t and not account for scr refresh
                    cue_stim_recall.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(cue_stim_recall, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'cue_stim_recall.started')
                    # update status
                    cue_stim_recall.status = STARTED
                    cue_stim_recall.setAutoDraw(True)
                
                # if cue_stim_recall is active this frame...
                if cue_stim_recall.status == STARTED:
                    # update params
                    pass
                
                # if cue_stim_recall is stopping this frame...
                if cue_stim_recall.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > cue_stim_recall.tStartRefresh + 3.0-frameTolerance:
                        # keep track of stop time/frame for later
                        cue_stim_recall.tStop = t  # not accounting for scr refresh
                        cue_stim_recall.tStopRefresh = tThisFlipGlobal  # on global time
                        cue_stim_recall.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'cue_stim_recall.stopped')
                        # update status
                        cue_stim_recall.status = FINISHED
                        cue_stim_recall.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    cued_recall.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in cued_recall.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "cued_recall" ---
            for thisComponent in cued_recall.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for cued_recall
            cued_recall.tStop = globalClock.getTime(format='float')
            cued_recall.tStopRefresh = tThisFlipGlobal
            thisExp.addData('cued_recall.stopped', cued_recall.tStop)
            # check responses
            if end_cued_recall.keys in ['', [], None]:  # No response was made
                end_cued_recall.keys = None
            test_trials.addData('end_cued_recall.keys',end_cued_recall.keys)
            if end_cued_recall.keys != None:  # we had a response
                test_trials.addData('end_cued_recall.rt', end_cued_recall.rt)
                test_trials.addData('end_cued_recall.duration', end_cued_recall.duration)
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if cued_recall.maxDurationReached:
                routineTimer.addTime(-cued_recall.maxDuration)
            elif cued_recall.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-3.000000)
            
            # --- Prepare to start Routine "recall_response" ---
            # create an object to store info about Routine recall_response
            recall_response = data.Routine(
                name='recall_response',
                components=[dash_stim_recall_resp, cue_stim_recall_resp, recall_stim_resp, recall_question, recall_reached],
            )
            recall_response.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            cue_stim_recall_resp.setText(this_cue)
            # create starting attributes for recall_reached
            recall_reached.keys = []
            recall_reached.rt = []
            _recall_reached_allKeys = []
            # allowedKeys looks like a variable, so make sure it exists locally
            if 'all_keys' in globals():
                all_keys = globals()['all_keys']
            # store start times for recall_response
            recall_response.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            recall_response.tStart = globalClock.getTime(format='float')
            recall_response.status = STARTED
            thisExp.addData('recall_response.started', recall_response.tStart)
            recall_response.maxDuration = None
            # keep track of which components have finished
            recall_responseComponents = recall_response.components
            for thisComponent in recall_response.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "recall_response" ---
            # if trial has changed, end Routine now
            if isinstance(test_trials, data.TrialHandler2) and thisTest_trial.thisN != test_trials.thisTrial.thisN:
                continueRoutine = False
            recall_response.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 1.5:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *dash_stim_recall_resp* updates
                
                # if dash_stim_recall_resp is starting this frame...
                if dash_stim_recall_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    dash_stim_recall_resp.frameNStart = frameN  # exact frame index
                    dash_stim_recall_resp.tStart = t  # local t and not account for scr refresh
                    dash_stim_recall_resp.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(dash_stim_recall_resp, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'dash_stim_recall_resp.started')
                    # update status
                    dash_stim_recall_resp.status = STARTED
                    dash_stim_recall_resp.setAutoDraw(True)
                
                # if dash_stim_recall_resp is active this frame...
                if dash_stim_recall_resp.status == STARTED:
                    # update params
                    pass
                
                # if dash_stim_recall_resp is stopping this frame...
                if dash_stim_recall_resp.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > dash_stim_recall_resp.tStartRefresh + 1.5-frameTolerance:
                        # keep track of stop time/frame for later
                        dash_stim_recall_resp.tStop = t  # not accounting for scr refresh
                        dash_stim_recall_resp.tStopRefresh = tThisFlipGlobal  # on global time
                        dash_stim_recall_resp.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'dash_stim_recall_resp.stopped')
                        # update status
                        dash_stim_recall_resp.status = FINISHED
                        dash_stim_recall_resp.setAutoDraw(False)
                
                # *cue_stim_recall_resp* updates
                
                # if cue_stim_recall_resp is starting this frame...
                if cue_stim_recall_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    cue_stim_recall_resp.frameNStart = frameN  # exact frame index
                    cue_stim_recall_resp.tStart = t  # local t and not account for scr refresh
                    cue_stim_recall_resp.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(cue_stim_recall_resp, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'cue_stim_recall_resp.started')
                    # update status
                    cue_stim_recall_resp.status = STARTED
                    cue_stim_recall_resp.setAutoDraw(True)
                
                # if cue_stim_recall_resp is active this frame...
                if cue_stim_recall_resp.status == STARTED:
                    # update params
                    pass
                
                # if cue_stim_recall_resp is stopping this frame...
                if cue_stim_recall_resp.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > cue_stim_recall_resp.tStartRefresh + 1.5-frameTolerance:
                        # keep track of stop time/frame for later
                        cue_stim_recall_resp.tStop = t  # not accounting for scr refresh
                        cue_stim_recall_resp.tStopRefresh = tThisFlipGlobal  # on global time
                        cue_stim_recall_resp.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'cue_stim_recall_resp.stopped')
                        # update status
                        cue_stim_recall_resp.status = FINISHED
                        cue_stim_recall_resp.setAutoDraw(False)
                
                # *recall_stim_resp* updates
                
                # if recall_stim_resp is starting this frame...
                if recall_stim_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    recall_stim_resp.frameNStart = frameN  # exact frame index
                    recall_stim_resp.tStart = t  # local t and not account for scr refresh
                    recall_stim_resp.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(recall_stim_resp, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'recall_stim_resp.started')
                    # update status
                    recall_stim_resp.status = STARTED
                    recall_stim_resp.setAutoDraw(True)
                
                # if recall_stim_resp is active this frame...
                if recall_stim_resp.status == STARTED:
                    # update params
                    pass
                
                # if recall_stim_resp is stopping this frame...
                if recall_stim_resp.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > recall_stim_resp.tStartRefresh + 1.5-frameTolerance:
                        # keep track of stop time/frame for later
                        recall_stim_resp.tStop = t  # not accounting for scr refresh
                        recall_stim_resp.tStopRefresh = tThisFlipGlobal  # on global time
                        recall_stim_resp.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'recall_stim_resp.stopped')
                        # update status
                        recall_stim_resp.status = FINISHED
                        recall_stim_resp.setAutoDraw(False)
                
                # *recall_question* updates
                
                # if recall_question is starting this frame...
                if recall_question.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    recall_question.frameNStart = frameN  # exact frame index
                    recall_question.tStart = t  # local t and not account for scr refresh
                    recall_question.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(recall_question, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'recall_question.started')
                    # update status
                    recall_question.status = STARTED
                    recall_question.setAutoDraw(True)
                
                # if recall_question is active this frame...
                if recall_question.status == STARTED:
                    # update params
                    pass
                
                # if recall_question is stopping this frame...
                if recall_question.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > recall_question.tStartRefresh + 1.5-frameTolerance:
                        # keep track of stop time/frame for later
                        recall_question.tStop = t  # not accounting for scr refresh
                        recall_question.tStopRefresh = tThisFlipGlobal  # on global time
                        recall_question.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'recall_question.stopped')
                        # update status
                        recall_question.status = FINISHED
                        recall_question.setAutoDraw(False)
                
                # *recall_reached* updates
                waitOnFlip = False
                
                # if recall_reached is starting this frame...
                if recall_reached.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    recall_reached.frameNStart = frameN  # exact frame index
                    recall_reached.tStart = t  # local t and not account for scr refresh
                    recall_reached.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(recall_reached, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'recall_reached.started')
                    # update status
                    recall_reached.status = STARTED
                    # allowed keys looks like a variable named `all_keys`
                    if not type(all_keys) in [list, tuple, np.ndarray]:
                        if not isinstance(all_keys, str):
                            all_keys = str(all_keys)
                        elif not ',' in all_keys:
                            all_keys = (all_keys,)
                        else:
                            all_keys = eval(all_keys)
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(recall_reached.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(recall_reached.clearEvents, eventType='keyboard')  # clear events on next screen flip
                
                # if recall_reached is stopping this frame...
                if recall_reached.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > recall_reached.tStartRefresh + 1.5-frameTolerance:
                        # keep track of stop time/frame for later
                        recall_reached.tStop = t  # not accounting for scr refresh
                        recall_reached.tStopRefresh = tThisFlipGlobal  # on global time
                        recall_reached.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'recall_reached.stopped')
                        # update status
                        recall_reached.status = FINISHED
                        recall_reached.status = FINISHED
                if recall_reached.status == STARTED and not waitOnFlip:
                    theseKeys = recall_reached.getKeys(keyList=list(all_keys), ignoreKeys=["escape"], waitRelease=False)
                    _recall_reached_allKeys.extend(theseKeys)
                    if len(_recall_reached_allKeys):
                        recall_reached.keys = _recall_reached_allKeys[-1].name  # just the last key pressed
                        recall_reached.rt = _recall_reached_allKeys[-1].rt
                        recall_reached.duration = _recall_reached_allKeys[-1].duration
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    recall_response.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in recall_response.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "recall_response" ---
            for thisComponent in recall_response.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for recall_response
            recall_response.tStop = globalClock.getTime(format='float')
            recall_response.tStopRefresh = tThisFlipGlobal
            thisExp.addData('recall_response.stopped', recall_response.tStop)
            # check responses
            if recall_reached.keys in ['', [], None]:  # No response was made
                recall_reached.keys = None
            test_trials.addData('recall_reached.keys',recall_reached.keys)
            if recall_reached.keys != None:  # we had a response
                test_trials.addData('recall_reached.rt', recall_reached.rt)
                test_trials.addData('recall_reached.duration', recall_reached.duration)
            # Run 'End Routine' code from skip_recall_select
            skip_letters = 0
            trial_dur = letter_dur
            if recall_reached.keys == succ_recall_key:
                trial_dur = letter_dur
            else:
                trial_dur = 0.00
                skip_letters = 1
            
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if recall_response.maxDurationReached:
                routineTimer.addTime(-recall_response.maxDuration)
            elif recall_response.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-1.500000)
            
            # --- Prepare to start Routine "recall_select" ---
            # create an object to store info about Routine recall_select
            recall_select = data.Routine(
                name='recall_select',
                components=[dash_stim_recall_select, cue_stim_recall_select, recall_stim_select, recall_question_select, recall_selection],
            )
            recall_select.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from letter_choice
            recall_select_text = ""
            
            random_letters = rnd.sample("abcdefghijklmnoprstuvz", k=3)
            target_letter = this_target[-1]
            while target_letter in random_letters:
                random_letters = rnd.sample("abcdefghijklmnoprstuvz", k=3)
            
            
            random_letters.append(target_letter)
            
            letters = rnd.sample(random_letters, len(random_letters))
            thisExp.addData('letter_options', letters)
            
            recall_select_text += "1) " + letters[0] + "        "
            recall_select_text += "2) " + letters[1] + "        "
            recall_select_text += "3) " + letters[2] + "        "
            recall_select_text += "4) " + letters[3] 
            
            cue_stim_recall_select.setText(this_cue)
            recall_question_select.setText(recall_select_text)
            # create starting attributes for recall_selection
            recall_selection.keys = []
            recall_selection.rt = []
            _recall_selection_allKeys = []
            # allowedKeys looks like a variable, so make sure it exists locally
            if 'all_keys' in globals():
                all_keys = globals()['all_keys']
            # store start times for recall_select
            recall_select.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            recall_select.tStart = globalClock.getTime(format='float')
            recall_select.status = STARTED
            thisExp.addData('recall_select.started', recall_select.tStart)
            recall_select.maxDuration = None
            # skip Routine recall_select if its 'Skip if' condition is True
            recall_select.skipped = continueRoutine and not (skip_letters >= 1)
            continueRoutine = recall_select.skipped
            # keep track of which components have finished
            recall_selectComponents = recall_select.components
            for thisComponent in recall_select.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "recall_select" ---
            # if trial has changed, end Routine now
            if isinstance(test_trials, data.TrialHandler2) and thisTest_trial.thisN != test_trials.thisTrial.thisN:
                continueRoutine = False
            recall_select.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *dash_stim_recall_select* updates
                
                # if dash_stim_recall_select is starting this frame...
                if dash_stim_recall_select.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    dash_stim_recall_select.frameNStart = frameN  # exact frame index
                    dash_stim_recall_select.tStart = t  # local t and not account for scr refresh
                    dash_stim_recall_select.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(dash_stim_recall_select, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'dash_stim_recall_select.started')
                    # update status
                    dash_stim_recall_select.status = STARTED
                    dash_stim_recall_select.setAutoDraw(True)
                
                # if dash_stim_recall_select is active this frame...
                if dash_stim_recall_select.status == STARTED:
                    # update params
                    pass
                
                # if dash_stim_recall_select is stopping this frame...
                if dash_stim_recall_select.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > dash_stim_recall_select.tStartRefresh + trial_dur-frameTolerance:
                        # keep track of stop time/frame for later
                        dash_stim_recall_select.tStop = t  # not accounting for scr refresh
                        dash_stim_recall_select.tStopRefresh = tThisFlipGlobal  # on global time
                        dash_stim_recall_select.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'dash_stim_recall_select.stopped')
                        # update status
                        dash_stim_recall_select.status = FINISHED
                        dash_stim_recall_select.setAutoDraw(False)
                
                # *cue_stim_recall_select* updates
                
                # if cue_stim_recall_select is starting this frame...
                if cue_stim_recall_select.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    cue_stim_recall_select.frameNStart = frameN  # exact frame index
                    cue_stim_recall_select.tStart = t  # local t and not account for scr refresh
                    cue_stim_recall_select.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(cue_stim_recall_select, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'cue_stim_recall_select.started')
                    # update status
                    cue_stim_recall_select.status = STARTED
                    cue_stim_recall_select.setAutoDraw(True)
                
                # if cue_stim_recall_select is active this frame...
                if cue_stim_recall_select.status == STARTED:
                    # update params
                    pass
                
                # if cue_stim_recall_select is stopping this frame...
                if cue_stim_recall_select.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > cue_stim_recall_select.tStartRefresh + trial_dur-frameTolerance:
                        # keep track of stop time/frame for later
                        cue_stim_recall_select.tStop = t  # not accounting for scr refresh
                        cue_stim_recall_select.tStopRefresh = tThisFlipGlobal  # on global time
                        cue_stim_recall_select.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'cue_stim_recall_select.stopped')
                        # update status
                        cue_stim_recall_select.status = FINISHED
                        cue_stim_recall_select.setAutoDraw(False)
                
                # *recall_stim_select* updates
                
                # if recall_stim_select is starting this frame...
                if recall_stim_select.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    recall_stim_select.frameNStart = frameN  # exact frame index
                    recall_stim_select.tStart = t  # local t and not account for scr refresh
                    recall_stim_select.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(recall_stim_select, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'recall_stim_select.started')
                    # update status
                    recall_stim_select.status = STARTED
                    recall_stim_select.setAutoDraw(True)
                
                # if recall_stim_select is active this frame...
                if recall_stim_select.status == STARTED:
                    # update params
                    pass
                
                # if recall_stim_select is stopping this frame...
                if recall_stim_select.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > recall_stim_select.tStartRefresh + trial_dur-frameTolerance:
                        # keep track of stop time/frame for later
                        recall_stim_select.tStop = t  # not accounting for scr refresh
                        recall_stim_select.tStopRefresh = tThisFlipGlobal  # on global time
                        recall_stim_select.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'recall_stim_select.stopped')
                        # update status
                        recall_stim_select.status = FINISHED
                        recall_stim_select.setAutoDraw(False)
                
                # *recall_question_select* updates
                
                # if recall_question_select is starting this frame...
                if recall_question_select.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    recall_question_select.frameNStart = frameN  # exact frame index
                    recall_question_select.tStart = t  # local t and not account for scr refresh
                    recall_question_select.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(recall_question_select, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'recall_question_select.started')
                    # update status
                    recall_question_select.status = STARTED
                    recall_question_select.setAutoDraw(True)
                
                # if recall_question_select is active this frame...
                if recall_question_select.status == STARTED:
                    # update params
                    pass
                
                # if recall_question_select is stopping this frame...
                if recall_question_select.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > recall_question_select.tStartRefresh + trial_dur-frameTolerance:
                        # keep track of stop time/frame for later
                        recall_question_select.tStop = t  # not accounting for scr refresh
                        recall_question_select.tStopRefresh = tThisFlipGlobal  # on global time
                        recall_question_select.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'recall_question_select.stopped')
                        # update status
                        recall_question_select.status = FINISHED
                        recall_question_select.setAutoDraw(False)
                
                # *recall_selection* updates
                waitOnFlip = False
                
                # if recall_selection is starting this frame...
                if recall_selection.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    recall_selection.frameNStart = frameN  # exact frame index
                    recall_selection.tStart = t  # local t and not account for scr refresh
                    recall_selection.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(recall_selection, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'recall_selection.started')
                    # update status
                    recall_selection.status = STARTED
                    # allowed keys looks like a variable named `all_keys`
                    if not type(all_keys) in [list, tuple, np.ndarray]:
                        if not isinstance(all_keys, str):
                            all_keys = str(all_keys)
                        elif not ',' in all_keys:
                            all_keys = (all_keys,)
                        else:
                            all_keys = eval(all_keys)
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(recall_selection.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(recall_selection.clearEvents, eventType='keyboard')  # clear events on next screen flip
                
                # if recall_selection is stopping this frame...
                if recall_selection.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > recall_selection.tStartRefresh + trial_dur-frameTolerance:
                        # keep track of stop time/frame for later
                        recall_selection.tStop = t  # not accounting for scr refresh
                        recall_selection.tStopRefresh = tThisFlipGlobal  # on global time
                        recall_selection.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'recall_selection.stopped')
                        # update status
                        recall_selection.status = FINISHED
                        recall_selection.status = FINISHED
                if recall_selection.status == STARTED and not waitOnFlip:
                    theseKeys = recall_selection.getKeys(keyList=list(all_keys), ignoreKeys=["escape"], waitRelease=False)
                    _recall_selection_allKeys.extend(theseKeys)
                    if len(_recall_selection_allKeys):
                        recall_selection.keys = _recall_selection_allKeys[-1].name  # just the last key pressed
                        recall_selection.rt = _recall_selection_allKeys[-1].rt
                        recall_selection.duration = _recall_selection_allKeys[-1].duration
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    recall_select.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in recall_select.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "recall_select" ---
            for thisComponent in recall_select.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for recall_select
            recall_select.tStop = globalClock.getTime(format='float')
            recall_select.tStopRefresh = tThisFlipGlobal
            thisExp.addData('recall_select.stopped', recall_select.tStop)
            # Run 'End Routine' code from letter_choice
            this_choice = ""
            correct_choice = 0
            if skip_letters  <= 1:
                if len(recall_selection.keys) > 0:
                    this_key = int(recall_selection.keys[0]) - 2
                    this_choice = letters[this_key]
            
                    if this_choice == target_letter:
                        correct_choice = 1
                        if (run_counter == 1) or (run_counter == 2):
                            ncorrect = ncorrect + 1
                            
                            if cue_types[this_cue] == "Read":
                                correct_read = correct_read + 1
                            elif cue_types[this_cue] == "Guess":
                                correct_guess = correct_guess + 1
            
            thisExp.addData('correct_choice', correct_choice)
            # check responses
            if recall_selection.keys in ['', [], None]:  # No response was made
                recall_selection.keys = None
            test_trials.addData('recall_selection.keys',recall_selection.keys)
            if recall_selection.keys != None:  # we had a response
                test_trials.addData('recall_selection.rt', recall_selection.rt)
                test_trials.addData('recall_selection.duration', recall_selection.duration)
            # the Routine "recall_select" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "iti_task" ---
            # create an object to store info about Routine iti_task
            iti_task = data.Routine(
                name='iti_task',
                components=[iti_cross],
            )
            iti_task.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # store start times for iti_task
            iti_task.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            iti_task.tStart = globalClock.getTime(format='float')
            iti_task.status = STARTED
            thisExp.addData('iti_task.started', iti_task.tStart)
            iti_task.maxDuration = None
            # keep track of which components have finished
            iti_taskComponents = iti_task.components
            for thisComponent in iti_task.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "iti_task" ---
            # if trial has changed, end Routine now
            if isinstance(test_trials, data.TrialHandler2) and thisTest_trial.thisN != test_trials.thisTrial.thisN:
                continueRoutine = False
            iti_task.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *iti_cross* updates
                
                # if iti_cross is starting this frame...
                if iti_cross.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    iti_cross.frameNStart = frameN  # exact frame index
                    iti_cross.tStart = t  # local t and not account for scr refresh
                    iti_cross.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(iti_cross, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'iti_cross.started')
                    # update status
                    iti_cross.status = STARTED
                    iti_cross.setAutoDraw(True)
                
                # if iti_cross is active this frame...
                if iti_cross.status == STARTED:
                    # update params
                    pass
                
                # if iti_cross is stopping this frame...
                if iti_cross.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > iti_cross.tStartRefresh + this_iti-frameTolerance:
                        # keep track of stop time/frame for later
                        iti_cross.tStop = t  # not accounting for scr refresh
                        iti_cross.tStopRefresh = tThisFlipGlobal  # on global time
                        iti_cross.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'iti_cross.stopped')
                        # update status
                        iti_cross.status = FINISHED
                        iti_cross.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    iti_task.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in iti_task.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "iti_task" ---
            for thisComponent in iti_task.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for iti_task
            iti_task.tStop = globalClock.getTime(format='float')
            iti_task.tStopRefresh = tThisFlipGlobal
            thisExp.addData('iti_task.stopped', iti_task.tStop)
            # the Routine "iti_task" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            thisExp.nextEntry()
            
        # completed num_trials repeats of 'test_trials'
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        # --- Prepare to start Routine "blank" ---
        # create an object to store info about Routine blank
        blank = data.Routine(
            name='blank',
            components=[begin_end_run_cross, sound_1, sound_2],
        )
        blank.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        sound_1.setSound('bell.wav', secs=1.2, hamming=True)
        sound_1.setVolume(1.0, log=False)
        sound_1.seek(0)
        sound_2.setSound('end_call.wav', secs=1, hamming=True)
        sound_2.setVolume(1.0, log=False)
        sound_2.seek(0)
        # store start times for blank
        blank.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        blank.tStart = globalClock.getTime(format='float')
        blank.status = STARTED
        thisExp.addData('blank.started', blank.tStart)
        blank.maxDuration = None
        # keep track of which components have finished
        blankComponents = blank.components
        for thisComponent in blank.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "blank" ---
        # if trial has changed, end Routine now
        if isinstance(recall_block, data.TrialHandler2) and thisRecall_block.thisN != recall_block.thisTrial.thisN:
            continueRoutine = False
        blank.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 12.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *begin_end_run_cross* updates
            
            # if begin_end_run_cross is starting this frame...
            if begin_end_run_cross.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                begin_end_run_cross.frameNStart = frameN  # exact frame index
                begin_end_run_cross.tStart = t  # local t and not account for scr refresh
                begin_end_run_cross.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(begin_end_run_cross, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'begin_end_run_cross.started')
                # update status
                begin_end_run_cross.status = STARTED
                begin_end_run_cross.setAutoDraw(True)
            
            # if begin_end_run_cross is active this frame...
            if begin_end_run_cross.status == STARTED:
                # update params
                pass
            
            # if begin_end_run_cross is stopping this frame...
            if begin_end_run_cross.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > begin_end_run_cross.tStartRefresh + 12.0-frameTolerance:
                    # keep track of stop time/frame for later
                    begin_end_run_cross.tStop = t  # not accounting for scr refresh
                    begin_end_run_cross.tStopRefresh = tThisFlipGlobal  # on global time
                    begin_end_run_cross.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'begin_end_run_cross.stopped')
                    # update status
                    begin_end_run_cross.status = FINISHED
                    begin_end_run_cross.setAutoDraw(False)
            
            # *sound_1* updates
            
            # if sound_1 is starting this frame...
            if sound_1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                sound_1.frameNStart = frameN  # exact frame index
                sound_1.tStart = t  # local t and not account for scr refresh
                sound_1.tStartRefresh = tThisFlipGlobal  # on global time
                # add timestamp to datafile
                thisExp.addData('sound_1.started', tThisFlipGlobal)
                # update status
                sound_1.status = STARTED
                sound_1.play(when=win)  # sync with win flip
            
            # if sound_1 is stopping this frame...
            if sound_1.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > sound_1.tStartRefresh + 1.2-frameTolerance or sound_1.isFinished:
                    # keep track of stop time/frame for later
                    sound_1.tStop = t  # not accounting for scr refresh
                    sound_1.tStopRefresh = tThisFlipGlobal  # on global time
                    sound_1.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'sound_1.stopped')
                    # update status
                    sound_1.status = FINISHED
                    sound_1.stop()
            
            # *sound_2* updates
            
            # if sound_2 is starting this frame...
            if sound_2.status == NOT_STARTED and tThisFlip >= 10.0-frameTolerance:
                # keep track of start time/frame for later
                sound_2.frameNStart = frameN  # exact frame index
                sound_2.tStart = t  # local t and not account for scr refresh
                sound_2.tStartRefresh = tThisFlipGlobal  # on global time
                # add timestamp to datafile
                thisExp.addData('sound_2.started', tThisFlipGlobal)
                # update status
                sound_2.status = STARTED
                sound_2.play(when=win)  # sync with win flip
            
            # if sound_2 is stopping this frame...
            if sound_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > sound_2.tStartRefresh + 1-frameTolerance or sound_2.isFinished:
                    # keep track of stop time/frame for later
                    sound_2.tStop = t  # not accounting for scr refresh
                    sound_2.tStopRefresh = tThisFlipGlobal  # on global time
                    sound_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'sound_2.stopped')
                    # update status
                    sound_2.status = FINISHED
                    sound_2.stop()
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[sound_1, sound_2]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                blank.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in blank.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "blank" ---
        for thisComponent in blank.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for blank
        blank.tStop = globalClock.getTime(format='float')
        blank.tStopRefresh = tThisFlipGlobal
        thisExp.addData('blank.stopped', blank.tStop)
        sound_1.pause()  # ensure sound has stopped at end of Routine
        sound_2.pause()  # ensure sound has stopped at end of Routine
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if blank.maxDurationReached:
            routineTimer.addTime(-blank.maxDuration)
        elif blank.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-12.000000)
        
        # --- Prepare to start Routine "task_break" ---
        # create an object to store info about Routine task_break
        task_break = data.Routine(
            name='task_break',
            components=[task_break_text, task_break_resp],
        )
        task_break.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # create starting attributes for task_break_resp
        task_break_resp.keys = []
        task_break_resp.rt = []
        _task_break_resp_allKeys = []
        # allowedKeys looks like a variable, so make sure it exists locally
        if 'all_keys' in globals():
            all_keys = globals()['all_keys']
        # store start times for task_break
        task_break.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        task_break.tStart = globalClock.getTime(format='float')
        task_break.status = STARTED
        thisExp.addData('task_break.started', task_break.tStart)
        task_break.maxDuration = None
        # skip Routine task_break if its 'Skip if' condition is True
        task_break.skipped = continueRoutine and not (run_counter >= 2)
        continueRoutine = task_break.skipped
        # keep track of which components have finished
        task_breakComponents = task_break.components
        for thisComponent in task_break.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "task_break" ---
        # if trial has changed, end Routine now
        if isinstance(recall_block, data.TrialHandler2) and thisRecall_block.thisN != recall_block.thisTrial.thisN:
            continueRoutine = False
        task_break.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *task_break_text* updates
            
            # if task_break_text is starting this frame...
            if task_break_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                task_break_text.frameNStart = frameN  # exact frame index
                task_break_text.tStart = t  # local t and not account for scr refresh
                task_break_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(task_break_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'task_break_text.started')
                # update status
                task_break_text.status = STARTED
                task_break_text.setAutoDraw(True)
            
            # if task_break_text is active this frame...
            if task_break_text.status == STARTED:
                # update params
                pass
            
            # *task_break_resp* updates
            waitOnFlip = False
            
            # if task_break_resp is starting this frame...
            if task_break_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                task_break_resp.frameNStart = frameN  # exact frame index
                task_break_resp.tStart = t  # local t and not account for scr refresh
                task_break_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(task_break_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'task_break_resp.started')
                # update status
                task_break_resp.status = STARTED
                # allowed keys looks like a variable named `all_keys`
                if not type(all_keys) in [list, tuple, np.ndarray]:
                    if not isinstance(all_keys, str):
                        all_keys = str(all_keys)
                    elif not ',' in all_keys:
                        all_keys = (all_keys,)
                    else:
                        all_keys = eval(all_keys)
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(task_break_resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(task_break_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if task_break_resp.status == STARTED and not waitOnFlip:
                theseKeys = task_break_resp.getKeys(keyList=list(all_keys), ignoreKeys=["escape"], waitRelease=False)
                _task_break_resp_allKeys.extend(theseKeys)
                if len(_task_break_resp_allKeys):
                    task_break_resp.keys = _task_break_resp_allKeys[-1].name  # just the last key pressed
                    task_break_resp.rt = _task_break_resp_allKeys[-1].rt
                    task_break_resp.duration = _task_break_resp_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                task_break.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in task_break.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "task_break" ---
        for thisComponent in task_break.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for task_break
        task_break.tStop = globalClock.getTime(format='float')
        task_break.tStopRefresh = tThisFlipGlobal
        thisExp.addData('task_break.stopped', task_break.tStop)
        # check responses
        if task_break_resp.keys in ['', [], None]:  # No response was made
            task_break_resp.keys = None
        recall_block.addData('task_break_resp.keys',task_break_resp.keys)
        if task_break_resp.keys != None:  # we had a response
            recall_block.addData('task_break_resp.rt', task_break_resp.rt)
            recall_block.addData('task_break_resp.duration', task_break_resp.duration)
        # Run 'End Routine' code from task_reset_counter
        run_counter = run_counter + 1
        # the Routine "task_break" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
    # completed 2.0 repeats of 'recall_block'
    
    
    # --- Prepare to start Routine "end_part2" ---
    # create an object to store info about Routine end_part2
    end_part2 = data.Routine(
        name='end_part2',
        components=[end_part2_text],
    )
    end_part2.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # Run 'Begin Routine' code from print_performance
    print("Correctly selected: ", ncorrect)
    print("Correct pretested: ", correct_guess)
    print("Correct read: ", correct_read)
    # store start times for end_part2
    end_part2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    end_part2.tStart = globalClock.getTime(format='float')
    end_part2.status = STARTED
    thisExp.addData('end_part2.started', end_part2.tStart)
    end_part2.maxDuration = None
    # keep track of which components have finished
    end_part2Components = end_part2.components
    for thisComponent in end_part2.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "end_part2" ---
    end_part2.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 30.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *end_part2_text* updates
        
        # if end_part2_text is starting this frame...
        if end_part2_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            end_part2_text.frameNStart = frameN  # exact frame index
            end_part2_text.tStart = t  # local t and not account for scr refresh
            end_part2_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(end_part2_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'end_part2_text.started')
            # update status
            end_part2_text.status = STARTED
            end_part2_text.setAutoDraw(True)
        
        # if end_part2_text is active this frame...
        if end_part2_text.status == STARTED:
            # update params
            pass
        
        # if end_part2_text is stopping this frame...
        if end_part2_text.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > end_part2_text.tStartRefresh + 30-frameTolerance:
                # keep track of stop time/frame for later
                end_part2_text.tStop = t  # not accounting for scr refresh
                end_part2_text.tStopRefresh = tThisFlipGlobal  # on global time
                end_part2_text.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'end_part2_text.stopped')
                # update status
                end_part2_text.status = FINISHED
                end_part2_text.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            end_part2.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in end_part2.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "end_part2" ---
    for thisComponent in end_part2.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for end_part2
    end_part2.tStop = globalClock.getTime(format='float')
    end_part2.tStopRefresh = tThisFlipGlobal
    thisExp.addData('end_part2.stopped', end_part2.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if end_part2.maxDurationReached:
        routineTimer.addTime(-end_part2.maxDuration)
    elif end_part2.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-30.000000)
    thisExp.nextEntry()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
