sudo() {
    if [ "$1" = "apt" ] && ([ "$2" = "upgrade" ] || [ "$2" = "dist-upgrade" ] || [ "$2" = "full-upgrade" ]); then
        echo "🚫 'sudo apt $2' is deprecated! Please contact Galaxea AI for help."
        return 1
    else
        command sudo "$@"
    fi
}

apt() {
   if [ "$1" = "dist-upgrade" ] || [ "$1" = "full-upgrade" ] || [ "$1" = "upgrade" ]; then
       echo "🚫 'apt $1' is deprecated! Please contact Galaxea AI for help."
       return 1
   else
       command apt "$@"
   fi
}