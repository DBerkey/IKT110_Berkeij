$(document).ready(function () {
    let radiant_pool = []
    let dire_pool = []
    let ban_pool = []

    function logPools() {
        console.log("radiant: " + radiant_pool + "\ndire: " + dire_pool + "\nbans: " + ban_pool)
    }

    /*
    ======================================
    PICK EVENTS
    ======================================
    */

    $(".hero-select-left").click(function () {
        let hero_id = $(this).data("heroid");
        add_to_pool(radiant_pool, dire_pool, ban_pool, hero_id)
        update_images("radiant", radiant_pool)
        logPools()

    })

    $(".hero-select-right").click(function () {
        let hero_id = $(this).data("heroid");
        add_to_pool(dire_pool, radiant_pool, ban_pool, hero_id)
        update_images("dire", dire_pool)
        logPools()
    })

    $(".hero-select-ban").click(function () {
        let hero_id = $(this).data("heroid");
        add_to_ban_pool(ban_pool, radiant_pool, dire_pool, hero_id)
        update_bans(ban_pool)
        logPools()
    })

    $(".hero-pool-radiant").click(function () {
        let hero_id = $(this).data("heroid");
        if (hero_id !== -1) {
            remove_from_pool(radiant_pool, hero_id)
            refresh_images()
            update_images("radiant", radiant_pool)
            update_images("dire", dire_pool)
            update_bans(ban_pool)

            logPools()
        }
    })

    $(".hero-pool-dire").click(function () {
        let hero_id = $(this).data("heroid");
        if (hero_id !== -1) {
            remove_from_pool(dire_pool, hero_id)
            refresh_images()
            update_images("dire", dire_pool)
            update_images("radiant", radiant_pool)
            update_bans(ban_pool)

            logPools()

        }
    })

    $(".hero-pool-ban").click(function () {
        let hero_id = $(this).data("heroid");
        if (hero_id !== -1) {
            remove_from_pool(ban_pool, hero_id)
            refresh_images()
            update_images("radiant", radiant_pool)
            update_images("dire", dire_pool)
            update_bans(ban_pool)
            logPools()
        }
    })

    /*
    ======================================
    BUTTON EVENTS
    ======================================
    */

    let output_box = $("#outputOracle")

    $("#suggestBtn1").click(function () {
        $.ajax({
            type: "POST",
            url: "/suggest1",
            contentType: "application/json; charset=utf-8",
            data: JSON.stringify({
                "radiant": radiant_pool,
                "dire": dire_pool
            }),
            dataType: "json",
            success: function (data) {
                let pretty_data = JSON.stringify(data, undefined, 4)
                output_box.text(pretty_data)

            }
        });
    })

    $("#suggestBtn2").click(function () {
        $.ajax({
            type: "POST",
            url: "/suggest2",
            contentType: "application/json; charset=utf-8",
            data: JSON.stringify({
                "radiant": radiant_pool,
                "dire": dire_pool
            }),
            dataType: "json",
            success: function (data) {
                let pretty_data = JSON.stringify(data, undefined, 4)
                output_box.text(pretty_data)

            }
        });
    })


    $("#suggestBtn3").click(function () {
        if (radiant_pool.length == 5 && dire_pool.length == 5) {
            $.ajax({
                type: "POST",
                url: "/suggest3",
                contentType: "application/json; charset=utf-8",
                data: JSON.stringify({
                    "radiant": radiant_pool,
                    "dire": dire_pool,
                    "bans": ban_pool
                }),
                dataType: "html",
                success: function (data) {
                    output_box.html(data)
                }
            });
        } else {
            output_box.text("Please select 5 heroes for both teams.");
        }
    })


});

function refresh_images() {
    $(".hero-pool").each(function () {
        $(this).attr("src", "/static/img/placeholder.png")
        $(this).data("heroid", -1);

    })
}

function update_images(faction, pool) {
    let imgs = $(`.hero-pool-${faction}`)

    for (const [i, hero_id] of pool.entries()) {
        $(imgs[i]).attr("src", `/static/img/avatar-sb/${hero_id.toString()}.png`)
        $(imgs[i]).css("height", "60px")
        $(imgs[i]).data("heroid", hero_id)
    }

}

function update_bans(pool) {
    let imgs = $(".hero-pool-ban")

    imgs.each(function () {
        $(this).attr("src", "/static/img/placeholder.png")
        $(this).data("heroid", -1)
    })

    for (const [i, hero_id] of pool.entries()) {
        $(imgs[i]).attr("src", `/static/img/avatar-sb/${hero_id.toString()}.png`)
        $(imgs[i]).css("height", "60px")
        $(imgs[i]).data("heroid", hero_id)
    }
}

function add_to_pool(pool, other_pool, ban_pool, id) {
    if (!pool.includes(id) && !other_pool.includes(id) && !ban_pool.includes(id) && pool.length < 5) {
        pool.push(id)
    }
}

function add_to_ban_pool(ban_pool, radiant_pool, dire_pool, id) {
    if (!ban_pool.includes(id) && !radiant_pool.includes(id) && !dire_pool.includes(id) && ban_pool.length < 14) {
        ban_pool.push(id)
    }
}

function remove_from_pool(pool, id) {
    if (pool.length >= 0) {
        let idx = pool.indexOf(id)
        if (idx > -1) {
            pool.splice(idx, 1);
        }
    }
}


